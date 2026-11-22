// Generated straight-line numeric kernel for one validated symbolic plan.
// No device-side plan buffer. Original entries are read on first use,
// fill entries start at zero. Save only eliminated L factors and scaled RHS
// for back substitution; Schur updates use named private variables.
alias Z=vec2<f64>;
@group(0) @binding(0) var<storage,read> original:array<f64>;
@group(0) @binding(1) var<storage,read> rhs:array<f64>;
@group(0) @binding(2) var<storage,read_write> factors:array<f64>;
@group(0) @binding(3) var<storage,read_write> solution:array<f64>;
fn mul(a:Z,b:Z)->Z { return Z(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x); }
fn inverse(a:Z)->Z { let n=a.x*a.x+a.y*a.y; return Z(a.x/n,-a.y/n); }
fn read_a(i:u32)->Z { return Z(original[i],original[i+64u]); }
fn read_b(i:u32)->Z { return Z(rhs[i],rhs[i+64u]); }
fn read_m(i:u32)->Z { return Z(factors[i],factors[i+64u]); }
fn put_m(i:u32,v:Z){ factors[i]=v.x; factors[i+64u]=v.y; }
fn read_x(i:u32)->Z { return Z(solution[i],solution[i+64u]); }
fn put_x(i:u32,v:Z){ solution[i]=v.x; solution[i+64u]=v.y; }
@compute @workgroup_size(64)
fn solve(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) lane:u32){
    let mb=group.x*ENTRY_COUNTu*128u+lane;
    let xb=group.x*ROW_COUNTu*128u+lane;
    NUMERIC_BODY
}
