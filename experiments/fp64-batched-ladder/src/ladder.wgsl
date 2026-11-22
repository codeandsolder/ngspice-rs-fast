// One invocation per independent 64-node grounded conductance ladder.
// Workgroup tile: [node][64 systems]; adjacent lanes read adjacent f64s.
@group(0) @binding(0) var<storage, read> diagonal: array<f64>;
@group(0) @binding(1) var<storage, read> rhs: array<f64>;
@group(0) @binding(2) var<storage, read_write> reciprocal: array<f64>;
@group(0) @binding(3) var<storage, read_write> solution: array<f64>;

@compute @workgroup_size(64)
fn solve(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let base = group.x * 4096u + lane;
    var p = 0.0lf;
    var y = 0.0lf;
    for (var row = 0u; row < 64u; row++) {
        let i = base + row * 64u;
        p = 1.0lf / (diagonal[i] - p);
        y = (rhs[i] + y) * p;
        reciprocal[i] = p;
        solution[i] = y;
    }
    var x = 0.0lf;
    for (var row = 64u; row > 0u; row--) {
        let i = base + (row - 1u) * 64u;
        x = solution[i] + reciprocal[i] * x;
        solution[i] = x;
    }
}
