function [r, dw, w2] = adaline(x, w1, theta, n, t)
    dw = false(1);
    w2 = false(1);
    r = (w1.'*x+theta);
    sumOfAll = w1.'*x;
    if ~(t == 'f')
        dw = n*(t-sumOfAll)*x;
        w2 = dw + w1;
    end
end