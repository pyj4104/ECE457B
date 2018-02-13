function [ w, dw ] = percepCalcW( n, t, o, x, w )
    dw = n * (t - o) * x;
    w = w + dw;
end

