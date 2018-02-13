function [epochItr, w, dw] = CheckConvergence(limit, theta, n, sigDigit)
    x1 = [1, -2, 0, -1].';
    x2 = [0, 1.5, -0.5, -1].';
    x3 = [-1, 1, 0.5, -1].';
    x = [x1, x2, x3];
    w = [1, -1, 0, 0.5].';
    dw = ones(4,1);
    wb = w - ones(4,1);
    t1 = -1;
    t2 = -1;
    t3 = 1;
    t = [t1, t2, t3];
    epochItr = 0;
    while((epochItr < limit) && ...
        (sum(round(w, sigDigit) - round(wb, sigDigit)) ~= 0))
        for i = 1:3
            wb = w;
            [~, dw, w] = adaline(x(:, i), w, theta, n, t(i));
        end
        epochItr = epochItr + 1;
    end
end