function [X]=PMC_S(X0,M,p,s,ker,maxiter)
% Polynomial Matrix Completion for Missing Data Imputation and Transductive Learning
% Jicong Fan, Yuqian Zhang, and Madeleine Udell. AAAI2020.
% X0: incomplete matrix
% M: binary mask matrix
% p: p value of Schatten-p norm
% s: the lower bound of rank
% ker: kernel typ and parameters, Gaussian RBF (suggested) or polynomial kernel
% for rbf kernel, there is one more paraemter ker.c, can be 0.5,1,2,3, or 5
[d,n]=size(X0);
X=X0;
if strcmp(ker.type,'rbf') && isempty(ker.par)
    XX=sum(X.*X,1);
    D=repmat(XX,n,1) + repmat(XX',1,n) - 2*X'*X;
    if isempty(ker.c)
        ker.c=3;% default
    end
    ker.par=(mean(real(D(:).^0.5))*ker.c)^2;% gamma % default
end
if strcmp(ker.type,'poly') && isempty(ker.par)
    ker.par=[1 2];% [c d]% default
end
% disp(['Kernel type: ' ker.type '; parameter(s):' num2str(ker.par)])    
idx=find(M==0);
% simple initialization: column-row mean
for i=1:d
    Ar(i)=mean(X(i,find(M(i,:)==1)));
end
for i=1:n
    Ac(i)=mean(X(find(M(:,i)==1),i));
end
for i=1:d
    for j=1:n
        A(i,j)=(Ar(i)+Ac(j))/2;
    end
end
x=A(find(M==0));
if length(x)==1
    x=ones(sum(1-M(:)),1)*x;
end
%
V.X0=X0;
V.M=M;
V.p=p;
V.ker=ker;
V.r=s;
%%
opt.alpha=1e-4;
opt.maxiter=maxiter;
[J,x]=opt_Adam(@MC_Kernel_T_Function,x,V,opt);
%%
X(idx)=x;
end

function [f,g]=MC_Kernel_T_Function(x,V)
%% parameters
X0=V.X0;
M=V.M;
p=V.p;
r=V.r;
ker=V.ker;
clear V;
[d,n]=size(X0);
%% kernel
tmp=X0(:);
idx=find(M==0);
tmp(idx)=x;
X=reshape(tmp,d,n);
if strcmp(ker.type,'rbf')
    XX=sum(X.*X,1);
    D=repmat(XX,n,1) + repmat(XX',1,n) - 2*X'*X;
    K=exp(-D/2/ker.par); 
end
if strcmp(ker.type,'poly')
    K=(X'*X+ker.par(1)).^ker.par(2);
end
%% objective function and gradient
K(isnan(K)) = 0;
K(isinf(K)) = 1000;
[V,S]=svd(K);
Vr=V(:,1:r);
s=diag(S);
f=sum(s(r+1:end).^(p/2));
K1=V*diag((s+ones(n,1)*1e-7).^(p/2-1))*V';
K2=Vr*diag(s(1:r).^(p/2-1))*Vr';
gLgK=p/2*K1-p/2*K2;

% %%% show eigvalue of kernel matrix
% [~,S_k,~]=svd(K);
% S_k=diag(S_k).^0.5;
% tt=sum(S_k(6:end));
% cc=fopen('xxx_toy_3_60_NLMC_RBF_s.txt','a');
% fprintf(cc,'%f\n',S_k);
% fclose(cc);
%%%%%%%

if strcmp(ker.type,'rbf')
    T=gLgK.*K;
    gLgX=2/ker.par*(X*T-X.*repmat(sum(T),d,1));
%     gLgX=2/ker.par*(X*T-bsxfun(@times,X,sum(T)));
    g=gLgX(idx);% gradient
end

if strcmp(ker.type,'poly')
    T=gLgK;
    gLgX=2*ker.par(2)*X*(T.*(X'*X+ker.par(1)).^(ker.par(2)-1));
    g=gLgX(idx);
end

end 





