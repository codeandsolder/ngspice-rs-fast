"""Time-domain *exact-delay* passive coupled LC/RG scattering cascade.

The power-normalized modal lossless LC stages are pure per-mode delays.
Every static R/G segment has a real passive reciprocal scattering matrix.
No passive pi ladder, no early wavefront, no unstable full-length ABCD.

This research reference uses a fixed sample grid with linear interpolation
of accepted past waves. It does not mutate accepted history during trial
evaluation, but it is not yet wired into native simulator MNA/StateHistory.
"""
import bisect
import heapq
from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from modal_propagation import LosslessModal
from scattering_split import scattering_from_transfer, split_scattering
from split_delay import telegrapher


@dataclass
class WaveTrace:
    times: list
    samples: list
    first_possible: np.ndarray

    def sample(self, time, mode):
        # Do not linearly interpolate *across* an analytically known
        # wavefront. Otherwise a positive sample after a delayed front
        # incorrectly leaks backward before that front.
        if time < self.first_possible[mode]:
            return 0.
        if time < 0:
            return 0.
        if time > self.times[-1]:
            raise ValueError("future delayed read requires step truncation")
        ix = bisect.bisect_right(self.times, time)
        if ix == 0:
            raise ValueError("lost needed accepted prehistory")
        if ix == len(self.times):
            return self.samples[-1][mode]
        earlier = ix-1
        t0, t1 = self.times[earlier], self.times[ix]
        alpha = (time-t0)/(t1-t0)
        return (1-alpha)*self.samples[earlier][mode]+alpha*self.samples[ix][mode]

    def prune(self, first_needed):
        """Retain the last interpolation bracket at/before first_needed.

        All future accepted-step reads target times >= first_needed;
        dropping that bracketing sample would silently invent a slope.
        """
        index=bisect.bisect_right(self.times, first_needed)-1
        if index>0:
            del self.times[:index]
            del self.samples[:index]

    def accept(self,time,sample):
        if not np.all(np.isfinite(sample)):
            raise ValueError("nonfinite accepted wave")
        if time <= self.times[-1]:
            raise ValueError("nonmonotone accepted wave time")
        self.times.append(float(time))
        values=np.asarray(sample)
        if np.any(np.abs(values.imag)>1e-12):
            raise ValueError("nonreal accepted time-domain waves")
        self.samples.append(np.asarray(values.real,dtype=float).copy())


class PassiveDelayCascade:
    def __init__(self,model,length,sections,near_front=None,far_front=None,
                 prune_history=True):
        model.validate()
        self.length=float(length)
        self.sections=int(sections)
        self.prune_history=bool(prune_history)
        self.n=len(model.r)
        if self.sections < 1 or self.length <= 0:
            raise ValueError("invalid number of static RG cells or length")
        self.modal=LosslessModal(model.l,model.c)
        dx=self.length/self.sections
        self.half_delays=np.asarray(self.modal.delay_per_length*dx/2)
        self.full_delays=2*self.half_delays
        static=expm(telegrapher(model,0.,1.,"static")*dx)
        self.static=np.asarray(
            np.real_if_close(scattering_from_transfer(static,self.modal),tol=1000)
        )
        if np.iscomplexobj(self.static):
            raise ValueError("complex static R/G scatter in real transient")
        if not np.allclose(self.static,self.static.T,atol=1e-12):
            raise ValueError("reciprocity error in static loss scattering")
        contraction=np.linalg.eigvalsh(self.static.T@self.static).max()
        if contraction > 1.+1e-10:
            raise ValueError("active static scattering element")
        self.near_front=np.asarray(
            np.zeros(self.n) if near_front is None else near_front,dtype=float
        )
        self.far_front=np.asarray(
            np.zeros(self.n) if far_front is None else far_front,dtype=float
        )
        if self.near_front.shape != (self.n,) or self.far_front.shape != (self.n,):
            raise ValueError("external incident wavefront dimension mismatch")
        if np.any(self.near_front<0) or np.any(self.far_front<0):
            raise ValueError("source wavefront cannot precede simulation epoch")
        self.static_front=self._static_front_times()
        self.out_near=[
            WaveTrace([-1e-99,0.],[np.zeros(self.n),np.zeros(self.n)],
                self.static_front[i,:self.n])
            for i in range(self.sections)
        ]
        self.out_far=[
            WaveTrace([-1e-99,0.],[np.zeros(self.n),np.zeros(self.n)],
                self.static_front[i,self.n:])
            for i in range(self.sections)
        ]
        self.input_near=WaveTrace([-1e-99,0.],
            [np.zeros(self.n),np.zeros(self.n)],self.near_front)
        self.input_far=WaveTrace([-1e-99,0.],
            [np.zeros(self.n),np.zeros(self.n)],self.far_front)
        self.time=0.
        self.revision=0
        self._owner_token=object()
        self.energy_in=0.
        self.energy_out=0.

    def _static_front_times(self):
        """Dijkstra propagation through exact per-mode delay edges.

        Every static R/G coupling is an instantaneous scattering edge.
        Each between-static LC segment contributes *its exact mode delay*.
        Source starts may be infinite, denoting an unexcited boundary.
        The earliest possible wave is a graph shortest path. This is
        physical wavefront metadata, not post-hoc output clipping.
        """
        n=self.n
        cells=self.sections
        width=2*n
        count=cells*width
        edges=[[] for _ in range(count)]
        fronts=[float("inf")]*count
        for i in range(cells):
            for output in range(width):
                current=i*width+output
                for input_side in range(2):
                    for mode in range(n):
                        input_index=input_side*n+mode
                        if self.static[output,input_index]==0.:
                            continue
                        if input_side==0:
                            if i==0:
                                fronts[current]=min(fronts[current],
                                    float(self.near_front[mode]+self.half_delays[mode]))
                                continue
                            upstream=(i-1)*width+n+mode
                        else:
                            if i==cells-1:
                                fronts[current]=min(fronts[current],
                                    float(self.far_front[mode]+self.half_delays[mode]))
                                continue
                            upstream=(i+1)*width+mode
                        edges[upstream].append((current,float(self.full_delays[mode])))
        heap=[(front,id) for id,front in enumerate(fronts) if np.isfinite(front)]
        heapq.heapify(heap)
        while heap:
            t,node=heapq.heappop(heap)
            if t>fronts[node]:
                continue
            for next_node,delta in edges[node]:
                arrival=t+delta
                if arrival<fronts[next_node]:
                    fronts[next_node]=arrival
                    heapq.heappush(heap,(arrival,next_node))
        return np.asarray(fronts).reshape(cells,width)

    def _sample(self, trace, time, delays):
        return np.array([trace.sample(time-delay,k)
            for k,delay in enumerate(delays)])

    def trial(self,t,incident_near,incident_far):
        if not np.isfinite(t) or t<=self.time:
            raise ValueError("trial time must be finite and ahead of acceptance")
        dt=t-self.time
        if dt > np.min(self.half_delays)*(1+1e-12):
            raise ValueError("step larger than fastest half-cell propagation delay")
        incoming_near=np.array(incident_near,dtype=float,copy=True)
        incoming_far=np.array(incident_far,dtype=float,copy=True)
        if incoming_near.shape!=(self.n,) or incoming_far.shape!=(self.n,):
            raise ValueError("port wave dimension mismatch")
        if not np.all(np.isfinite(incoming_near)) or not np.all(np.isfinite(incoming_far)):
            raise ValueError("nonfinite port wave")
        outs_near=[]
        outs_far=[]
        for i in range(self.sections):
            aL=self._sample(self.input_near,t,self.half_delays) if i==0 else self._sample(self.out_far[i-1],t,self.full_delays)
            aR=self._sample(self.input_far,t,self.half_delays) if i==self.sections-1 else self._sample(self.out_near[i+1],t,self.full_delays)
            result=self.static@np.r_[aL,aR]
            outs_near.append(result[:self.n])
            outs_far.append(result[self.n:])
        outgoing_near=self._sample(self.out_near[0],t,self.half_delays)
        outgoing_far=self._sample(self.out_far[-1],t,self.half_delays)
        for item in outs_near+outs_far+[outgoing_near,outgoing_far]:
            if not np.all(np.isfinite(item)):
                raise ValueError("nonfinite trial wave scattering")
        return {
            "time":float(t),
            "base_revision":self.revision,
            "owner_token":self._owner_token,
            "incoming_near":incoming_near,
            "incoming_far":incoming_far,
            "outgoing_near":outgoing_near,
            "outgoing_far":outgoing_far,
            "static_out_near":outs_near,
            "static_out_far":outs_far,
        }

    def accept(self,trial):
        t=trial["time"]
        if (not np.isfinite(t) or t<=self.time
            or trial["base_revision"] != self.revision
            or trial["owner_token"] is not self._owner_token):
            raise ValueError("stale, foreign or invalid wave acceptance")
        if (len(trial["static_out_near"])!=self.sections
            or len(trial["static_out_far"])!=self.sections):
            raise ValueError("malformed number of static scatter output blocks")
        values=[
            trial["incoming_near"],trial["incoming_far"],
            trial["outgoing_near"],trial["outgoing_far"],
            *trial["static_out_near"],*trial["static_out_far"]
        ]
        for item in values:
            if np.shape(item)!=(self.n,) or not np.all(np.isfinite(item)):
                raise ValueError("malformed or nonfinite accepted wave")
        dt=t-self.time
        # Validate the entire transaction, including stored energy,
        # before mutating ANY accepted history. An error must leave
        # all traces and revision intact.
        add_in=dt*(np.dot(values[0],values[0])+np.dot(values[1],values[1]))
        add_out=dt*(np.dot(values[2],values[2])+np.dot(values[3],values[3]))
        next_energy_in=self.energy_in+add_in
        next_energy_out=self.energy_out+add_out
        if not np.isfinite(next_energy_in) or not np.isfinite(next_energy_out):
            raise ValueError("accepted wave energy overflow")
        # The inputs and all output buffers are validated and copied
        # above, so these writes cannot fail on partial sample contents.
        self.energy_in=next_energy_in
        self.energy_out=next_energy_out
        for i in range(self.sections):
            self.out_near[i].accept(t,trial["static_out_near"][i])
            self.out_far[i].accept(t,trial["static_out_far"][i])
        self.input_near.accept(t,trial["incoming_near"])
        self.input_far.accept(t,trial["incoming_far"])
        self.time=t
        self.revision+=1
        if self.prune_history:
            # The largest possible LC propagation delay between any
            # pair of static output/input ports determines a uniform
            # safe retention horizon. This keeps one interpolation
            # bracket below that horizon for every modal trace.
            cutoff=t-float(np.max(self.full_delays))
            for trace in self.out_near+self.out_far+[self.input_near,self.input_far]:
                trace.prune(cutoff)

    def history_sample_count(self):
        return sum(len(trace.times) for trace in
            self.out_near+self.out_far+[self.input_near,self.input_far])

    def advance(self,t,incident_near,incident_far):
        trial=self.trial(t,incident_near,incident_far)
        self.accept(trial)
        return trial

def run(model,length,sections,source,final_time,dt=None,prune_history=True):
    network=PassiveDelayCascade(model,length,sections,
        far_front=np.full(len(model.r),np.inf),
        prune_history=prune_history)
    if dt is None:
        dt=float(np.min(network.half_delays))*0.5
    nsteps=int(np.ceil(final_time/dt))
    times=np.linspace(0.,final_time,nsteps+1)
    incoming=[]
    returned=[]
    transmitted=[]
    for t in times[1:]:
        inc=np.asarray(source(t),float)
        trial=network.advance(t,inc,np.zeros(network.n))
        incoming.append(inc)
        returned.append(trial["outgoing_near"])
        transmitted.append(trial["outgoing_far"])
    return (times[1:],np.array(incoming),np.array(returned),
        np.array(transmitted),network)
