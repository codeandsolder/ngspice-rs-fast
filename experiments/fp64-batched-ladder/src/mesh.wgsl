// Lower-triangle sparse symmetric LDL^T on one independent system per lane.
// Shared symbolic tape includes fill-in and back-substitution; numerical
// workspace is private by lane. The real and imaginary lanes are contiguous.
@group(0) @binding(0) var<storage,read> original: array<f64>;
@group(0) @binding(1) var<storage,read> rhs: array<f64>;
@group(0) @binding(2) var<storage,read_write> factor: array<f64>;
@group(0) @binding(3) var<storage,read_write> x: array<f64>;
@group(0) @binding(4) var<storage,read> tape: array<vec4<u32>>;

fn mul(a:vec2<f64>, b:vec2<f64>)->vec2<f64>{
    return vec2<f64>(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x);
}
fn inverse(a:vec2<f64>)->vec2<f64>{
    let norm=a.x*a.x+a.y*a.y;
    return vec2<f64>(a.x/norm,-a.y/norm);
}
fn read_m(i:u32)->vec2<f64>{return vec2<f64>(factor[i],factor[i+64u]);}
fn write_m(i:u32,v:vec2<f64>){factor[i]=v.x;factor[i+64u]=v.y;}
fn read_x(i:u32)->vec2<f64>{return vec2<f64>(x[i],x[i+64u]);}
fn write_x(i:u32,v:vec2<f64>){x[i]=v.x;x[i+64u]=v.y;}

@compute @workgroup_size(64)
fn solve(@builtin(workgroup_id) group:vec3<u32>, @builtin(local_invocation_index) lane:u32){
    let header=tape[0];
    let mb=group.x*header.x*128u+lane;
    let xb=group.x*header.y*128u+lane;
    for(var m=0u;m<header.x;m++){
        let i=mb+m*128u;
        factor[i]=original[i];
        factor[i+64u]=original[i+64u];
    }
    for(var r=0u;r<header.y;r++){
        let i=xb+r*128u;
        x[i]=rhs[i];
        x[i+64u]=rhs[i+64u];
    }
    var inv=vec2<f64>(0.0lf,0.0lf);
    var pivot_rhs=vec2<f64>(0.0lf,0.0lf);
    var multiplier=vec2<f64>(0.0lf,0.0lf);
    for(var op=1u;op<header.z;op++){
        let item=tape[op];
        let a=mb+item.y*128u;
        let b=xb+item.z*128u;
        switch item.x {
            case 0u:{
                inv=inverse(read_m(a));
                pivot_rhs=read_x(b);
                write_x(b,mul(inv,pivot_rhs));
            }
            case 1u:{
                multiplier=mul(read_m(a),inv);
                write_x(b,read_x(b)-mul(multiplier,pivot_rhs));
            }
            case 2u:{
                write_m(a,read_m(a)-mul(multiplier,read_m(mb+item.z*128u)));
            }
            case 3u:{write_m(a,multiplier);}
            case 4u:{
                let c=xb+item.w*128u;
                write_x(c,read_x(c)-mul(read_m(a),read_x(b)));
            }
            default:{}
        }
    }
}
