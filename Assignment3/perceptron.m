function [ o, w, dw, theta, dTheta ] = perceptron( x, t, w, n, theta )
    [o] = percepCalcOut(w, x, theta);
    [w, dw] = percepCalcW(n, t, o, x, w);
    [theta, dTheta] = percepCalcTheta(n, t, o, theta);
end

