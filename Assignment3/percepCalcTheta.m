function [ theta, dTheta ] = percepCalcTheta( n, t, o, theta )
    dTheta = -n*(t-o);
    theta = theta + dTheta;
end
