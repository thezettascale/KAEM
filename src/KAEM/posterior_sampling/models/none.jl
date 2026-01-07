module NoEncoder_Model

export NoEncoder, init_no_encoder

using Lux, Random, ConfParser

using ..Utils

struct NoEncoder <: Lux.AbstractLuxLayer end

function init_no_encoder(
        conf::ConfParse,
        x_shape::Tuple{Vararg{Int}},
        rng::AbstractRNG,
    )
    return NoEncoder()
end

end
