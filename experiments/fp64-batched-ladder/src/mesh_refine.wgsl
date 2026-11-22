// Use the original FP64 operator for residuals, reuse FP32 factors. The
// bounded loop has an explicit componentwise backward error criterion.
// Nonfinite comparisons fail the criterion; host validates final solution
// and residual independently, so failure is never treated as success.
    for(var iteration=0u;iteration<5u;iteration++){
        var converged=true;
        for(var i=0u;i<header.y;i++){
            let offset=xb+i*128u;
            let original_rhs=b64(offset);
            var residual=original_rhs;
            var scale=abs(original_rhs.x)+abs(original_rhs.y);
            let row=plan[header.w+i];
            for(var j=row.x;j<row.y;j++){
                let entry=plan[j];
                let a=a64(mb+entry.y*128u);
                let x=x64(xb+entry.x*128u);
                residual-=mul64(a,x);
                // L1 norms avoid unnecessary square-root/divide operations.
                scale+=(abs(a.x)+abs(a.y))*(abs(x.x)+abs(x.y));
            }
            let error=abs(residual.x)+abs(residual.y);
            converged=converged && error<=1e-14lf*scale;
            put_w(offset,Z(residual));
        }
        if converged { break; }
        solve_factored(mb,xb,header.y);
        for(var i=0u;i<header.y;i++){
            let offset=xb+i*128u;
            put_x(offset,x64(offset)+D(w(offset)));
        }
    }
