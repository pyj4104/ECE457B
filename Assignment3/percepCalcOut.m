function [ o ] = percepCalcOut( w, x, theta )
    o = w.'*x-theta;
    if o < 0
        o = -1;
    elseif o == 0
        o = 0;
    else
        o = 1;
    end
end

