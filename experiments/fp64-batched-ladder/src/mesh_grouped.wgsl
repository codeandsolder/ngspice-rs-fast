// Shared grouped symbolic plan. Factor precision is specialized at pipeline
// construction; original coefficients and final solution are always FP64.
alias F = FACTOR_SCALAR;
alias Z = vec2<F>;
alias D = vec2<f64>;
@group(0) @binding(0) var<storage,read> original: array<f64>;
@group(0) @binding(1) var<storage,read> rhs: array<f64>;
@group(0) @binding(2) var<storage,read_write> factor: array<F>;
@group(0) @binding(3) var<storage,read_write> work: array<F>;
@group(0) @binding(4) var<storage,read_write> solution: array<f64>;
@group(0) @binding(5) var<storage,read> plan: array<vec4<u32>>;

fn mul(a: Z, b: Z) -> Z {
    return Z(a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x);
}
fn mul64(a: D, b: D) -> D {
    return D(a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x);
}
fn inverse(a: Z) -> Z {
    // The tested coefficients are moderate, nonzero and well conditioned.
    let norm = a.x*a.x+a.y*a.y;
    return Z(a.x/norm,-a.y/norm);
}
fn m(i: u32) -> Z { return Z(factor[i],factor[i+64u]); }
fn put_m(i: u32,v: Z) { factor[i]=v.x; factor[i+64u]=v.y; }
fn w(i: u32) -> Z { return Z(work[i],work[i+64u]); }
fn put_w(i: u32,v: Z) { work[i]=v.x; work[i+64u]=v.y; }
fn a64(i: u32) -> D { return D(original[i],original[i+64u]); }
fn b64(i: u32) -> D { return D(rhs[i],rhs[i+64u]); }
fn x64(i: u32) -> D { return D(solution[i],solution[i+64u]); }
fn put_x(i: u32,v: D) { solution[i]=v.x; solution[i+64u]=v.y; }
fn factorize(base: u32,n: u32) {
    for(var k=0u;k<n;k++){
        let col=plan[1u+k];
        let d=base+col.x*128u;
        let inv=inverse(m(d));
        put_m(d,inv);
        for(var j=col.y;j<col.z;j++){
            let edge=plan[j];
            let handle=base+edge.y*128u;
            let l=mul(m(handle),inv);
            for(var u=edge.z;u<edge.w;u++){
                let update=plan[u];
                let dest=base+update.x*128u;
                put_m(dest,m(dest)-mul(l,m(base+update.y*128u)));
            }
            put_m(handle,l);
        }
    }
}
// L y = b, D z = y, L^T x = z; retain factors across correction solves.
fn solve_factored(mb:u32,xb:u32,n:u32) {
    for(var k=0u;k<n;k++){
        let col=plan[1u+k];
        let offset=xb+k*128u;
        let y=w(offset);
        for(var j=col.y;j<col.z;j++){
            let edge=plan[j];
            let dest=xb+edge.x*128u;
            put_w(dest,w(dest)-mul(m(mb+edge.y*128u),y));
        }
        put_w(offset,mul(m(mb+col.x*128u),y));
    }
    for(var k=n;k>0u;k--){
        let row=k-1u;
        let col=plan[1u+row];
        let offset=xb+row*128u;
        var value=w(offset);
        for(var j=col.y;j<col.z;j++){
            let edge=plan[j];
            value-=mul(m(mb+edge.y*128u),w(xb+edge.x*128u));
        }
        put_w(offset,value);
    }
}
@compute @workgroup_size(64)
fn solve(@builtin(workgroup_id) group:vec3<u32>, @builtin(local_invocation_index) lane:u32){
    let header=plan[0];
    let mb=group.x*header.x*128u+lane;
    let xb=group.x*header.y*128u+lane;
    for(var i=0u;i<header.x;i++){
        let offset=mb+i*128u;
        // Fill locations have known zero input.
        if i<header.z { put_m(offset,Z(a64(offset))); }
        else { put_m(offset,Z(F(0),F(0))); }
    }
    factorize(mb,header.y);
    for(var i=0u;i<header.y;i++){
        let offset=xb+i*128u;
        put_w(offset,Z(b64(offset)));
    }
    solve_factored(mb,xb,header.y);
    for(var i=0u;i<header.y;i++){
        let offset=xb+i*128u;
        put_x(offset,D(w(offset)));
    }
    REFINEMENT_BODY
}
