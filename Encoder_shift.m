function [H]=Encoder_shift(p,shft)
    e=eye(p-1,p-1);
    P=zeros(p,p);
    P(1:p-1,2:p)=e;
    P(p,1)=1;
    [r,c]=size(shft);
    
    for i=1:r
        for j=1:c
            H((i-1)*p+1:p*i,(j-1)*p+1:p*j)=P^(shft(i,j));
        end
    end
end