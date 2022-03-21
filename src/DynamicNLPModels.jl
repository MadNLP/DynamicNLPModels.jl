module DynamicNLPModels

import NLPModels
import QuadraticModels

mutable struct LQDynData{VT,MT}
    N::Int
    nx::Int
    nu::Int
    
    A::MT
    B::MT
    Q::MT
    R::MT
    Qf::MT
    
    x0::VT
    xl::VT
    xu::VT
    ul::VT
    uu::VT
end

function LQDynData(
    N, x0, A, B, Q, R;
    Qf = Q,
    xl = (similar(x0) .= -Inf),
    xu = (similar(x0) .=  Inf),
    ul = (similar(x0,nu) .= -Inf),
    uu = (similar(x0,nu) .=  Inf)
    )
    # TODO
end

function QPData(dnlp::LQDynData{VT,MT}) where {VT,MT}
    # TODO
end

end # module
