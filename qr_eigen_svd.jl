MatrixType=Float32

function norm1(v)
    g=v'*v
    return sqrt(g)
end

function diagOne!(M)
    for i in 1:size(M,1)
        M[i,i]=MatrixType(1)
    end
end

function H(A,l)
    m=size(A,1)
    @views x=A[l:m,l]
    @views w=(zeros(MatrixType,m+1-l));w[1]=MatrixType(norm1(x))
    @views v=w-x
    i=zeros(MatrixType,m+1-l,m+1-l)
    diagOne!(i)
    return i-2*v*v'/(v'*v)
end

function H1(A)
    m=size(A,1)
    C1=H(A,1)
    for i in 2:m-1
        h=H(C1*A,i)
        @views C1[i:end,1:i-1]=h*C1[i:end,1:i-1]
        @views C1[i:end,i:end]=h*C1[i:end,i:end]
    end
    return C1',C1*A
end


function eigen1(M)
    m=size(M,1)
    q0=zeros(MatrixType,m,m)
    diagOne!(q0)
    q=q0
    @show typeof(q0)
    r=M
    for i in 1:5
        q0,r=H1(r*q0)
        @show sum(abs.(q*q'))-m
        q=q*q0
    end
    return r*q0,q
end

function diagm1(v)
    l=length(v)
    M=zeros(MatrixType,l,l)
    for i in 1:l
        M[i,i]=v[i]
    end
    return M
end

function svd1(A,r)
    m=size(A,1);n=size(A,2);
    values,vectors=eigen1(A'*A);
    lam=zeros(MatrixType,n);
    for i in 1:n
        lam[i]=abs(values[i,i])
    end
    V=reverse(vectors,dims=2);
    lambda=reverse(lam);
    U=zeros(MatrixType,m,r);
    sigma=zeros(MatrixType,r);
    temp=Vector{MatrixType}(undef,m);
    (for i in 1:r
        temp=A*V[:,i]
        U[:,i]=temp/norm1(temp)
        sigma[i]=sqrt(lambda[i])
    end)
    sigma=diagm1(sigma)
    Vt=V'[1:r,:]
    return (U,sigma,Vt)
end

l=512
A=rand(MatrixType,l,l)
(U,sigma,Vt)=svd1(A,l);

println(l)

@timev (U,sigma,Vt)=svd1(A,l);

println(sum(abs.(U*sigma*Vt-A)))


