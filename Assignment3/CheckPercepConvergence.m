function [DoesConverge, epochItr, w, theta] = CheckPercepConvergence(points, n, limit, w, theta)
    totMat = points;
    
    error = 1;
    
    epochItr = 0;
    while((epochItr < limit) && (error ~= 0))
        error = 0;
        for i = 1:length(totMat)
            [~, w, dw, theta, dTheta] = perceptron(totMat([1:2],i), totMat(3, i), w, n, ...
                theta);
            error = error + sum(abs(dw)) + dTheta;
        end
        epochItr = epochItr + 1;
    end
    
    if(epochItr < limit)
        DoesConverge = 'T';
    else
        DoesConverge = 'F';
    end
end