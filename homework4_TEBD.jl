
# Homework 4.   Yifan Tian

using TensorOperations
using PyPlot

#Question 1
n = 10
A = [zeros(1,2,1) for i = 1:n]
for i = 1:n
  A[i][1,iseven(i)? 2:1,1] = 1.0
end

#(a) use loops to sum over all indices
function normsq1(A)
  l = size(A)[1]
  (m1,n,m2) = size(A[1])
  mpsl = zeros(m1,m1,m2,m2)
  for u1 = 1:m1, d1 = 1:m1, s = 1:n, mu1= 1:m2, md1 = 1:m2
   mpsl[u1,d1,mu1,md1] += A[1][u1,s,mu1]*A[1][d1,s,md1]
  end
  for i = 2:l
    mpsr = zeros(m1,m1,m2,m2)
    for u1 = 1:m1, d1 = 1:m1, s = 1:n, mu1= 1:m2, md1 = 1:m2, mui = 1:m2, mdi = 1:m2
      mpsr[u1,mui,mdi,d1] += mpsl[u1,mu1,md1,d1]*A[i][mu1,s,mui]*A[i][md1,s,mdi]
    end
    mpsl = mpsr
  end
  return mpsl[1,1,1,1]
end

#(b) use matrix product to sum over all indices
function normsq2(A)
  l = size(A)[1]
  (m1,n,m2) = size(A[2])
  mps1 = reshape(A[1],m1*m2,n)*reshape(A[1],n,m1*m2)
  mps1 = reshape(mps1,m1*m1*m2,m2)
  for i = 2:l
    mps1d = mps1*reshape(A[i],m1,n*m2)
    mps2 = reshape(mps1d,m2,n*m1)*reshape(A[i],n*m1,m2)
    mps1 = reshape(mps2,m1,m1)
  end
  return mps1[1,1]
end

#(c) use tensor operation library to contract
function normsq3(psi::MPS)
  l = length(psi.A)
  A1 = psi.A[1]
  @tensor begin
    mpsl[u1,d1,mu1,md1] := A1[u1,s,mu1]*A1[d1,s,md1]
  end
  for i = 2:l
    A1 = psi.A[i]
    @tensor mpsl[u1,d1,mu1,md1] := mpsl[u1,d1,mu,md]*A1[mu,s,mu1]*A1[md,s,md1]
  end
  return mpsl[1,1,1,1]
end

mp = MPS(A,1)
@time normsq1(A)
@time normsq2(A)
@time normsq3(mp)


#Question 2
type MPS
  A
  oc::Int64
end

#=
function reconstruct(psi::MPS)
  l = length(psi.A)
  for i = 2:l
    A1 = psi.A[i]
    (ml,s,mr) = size(psi.A[i])
      mpsl = reshape(mpsl,a,ml)
      mps = mpsl*reshape(A1,ml,s*mr)
      mpsl = reshape(mps,a,ml)
  end
  mps = reshape(mpsl,ml,s,s,s,s,mr)
  return mps
end
=#

function moveto!(psi::MPS,i::Int64)
  if i > psi.oc
    for n = psi.oc:i-1
      (ml,s,mr) = size(psi.A[n])
      intmps = reshape(psi.A[n],ml*s,mr)
      intq = qr(intmps)[1]; intr = qr(intmps)[2]
       if ml*s >= mr
          psi.A[n] = reshape(intq,ml,s,mr)
       else
          psi.A[n] = reshape(intq,ml,s,ml*s)
       end
      (ml_r,s_r,mr_r) = size(psi.A[n+1])
      if ml*s >= ml_r
        psi.A[n+1] = reshape(intr*reshape(psi.A[n+1],ml_r,s_r*mr_r),ml_r,s_r,mr_r)
      else
        psi.A[n+1] = reshape(intr*reshape(psi.A[n+1],ml_r,s_r*mr_r),ml*s,s_r,mr_r)
      end
    end
  elseif i < psi.oc
    for n = psi.oc:-1:(i+1)
      (ml,s,mr) = size(psi.A[n])
      intmps = reshape(psi.A[n],mr*s,ml)      #tricky part
      intq = qr(intmps)[1]; intr = qr(intmps)[2]
      if mr*s >= ml
        psi.A[n] = reshape(transpose(intq),ml,s,mr)
      else
        psi.A[n] = reshape(transpose(intq),mr*s,s,mr)
      end
      (ml_l,s_l,mr_l) = size(psi.A[n-1])
      if mr*s > mr_l
        psi.A[n-1] = reshape(reshape(psi.A[n-1],ml_l*s_l,mr_l)*intr,ml_l,s_l,mr_l)
      else
        psi.A[n-1] = reshape(reshape(psi.A[n-1],ml_l*s_l,mr_l)*intr,ml_l,s_l,mr*s)
      end
    end
  end
  psi.oc = i
  return psi
end

function normsqmt(psi::MPS)
  #used to check moveto by the feature of OC
  (ml,s,mr) = size(psi.A[psi.oc])
  l = size(psi.A)[1]
  psi = moveto!(psi,l)
  A = psi.A[1]
  @tensor begin
    norm[u1,d1] := A[u1,s,m1]*A[d1,s,m1]
  end
  return norm[1,1]
end

normsqmt(mp)
normsq3(moveto!(mp,100))

#Question 3
function energybond(psi::MPS,i::Int64)
  l = size(psi.A)[1]
  psi = moveto!(psi,i)
  mptwo1 = psi.A[i]; mptwo2 = psi.A[i+1]
  @tensor begin
    szbond = scalar(mptwo1[m1,usz1,eu]*mptwo2[eu,usz2,m2]*Htwosite[usz1,usz2,dsz1,dsz2]*mptwo1[m1,dsz1,ed]*mptwo2[ed,dsz2,m2])
  end
  return szbond
end

function energy_oc(psi::MPS)
  #calculate heisenberg energy using energybond
  l = size(psi.A)[1]
  energy = 0
    for i = l-1:-1:1
      energy += energybond(psi,i)
    end
  mp_norm = normsq3(psi)
  return energy/mp_norm
end

function energy_contra(psi::MPS)
  #calculate heisenberg energy using contraction
  l = size(psi.A)[1]
  energy = 0
  for i = 1:l-1
    energy += energybond_contra(psi,i)
  end
  mp_norm = normsq3(psi)
  return energy/mp_norm
end

function energybond_contra(psi::MPS,si)
  #calculate heisenberg energybond using contraction
  l = length(psi.A)
  mptwo1 = psi.A[si]; mptwo2 = psi.A[si+1];
  if si>1
    A1 = psi.A[1]
    @tensor begin
      mpsl[u1,d1,mu1,md1] := A1[u1,s,mu1]*A1[d1,s,md1]
    end
    for i = 2:si-1
      A1 = psi.A[i]
      @tensor mpsl[u1,d1,mu1,md1] := mpsl[u1,d1,mu,md]*A1[mu,s,mu1]*A1[md,s,md1]
    end
    @tensor mpsl[u1,d1,mu1,md1] := mpsl[u1,d1,ul,dl]*mptwo1[ul,usz1,mu]*mptwo2[mu,usz2,mu1]*Htwosite[usz1,usz2,dsz1,dsz2]*mptwo1[dl,dsz1,md]*mptwo2[md,dsz2,md1]
  else
    @tensor mpsl[u1,d1,mu1,md1] := mptwo1[u1,usz1,mu]*mptwo2[mu,usz2,mu1]*Htwosite[usz1,usz2,dsz1,dsz2]*mptwo1[d1,dsz1,md]*mptwo2[md,dsz2,md1]
  end
  for i = si+2:l
    A1 = psi.A[i]
    @tensor mpsl[u1,d1,mu1,md1] := mpsl[u1,d1,mu,md]*A1[mu,s,mu1]*A1[md,s,md1]
  end
  return mpsl[1,1,1,1]
end


#Question 4
sz = Float64[0.5 0;0 -0.5]; sp = Float64[0 1;0 0]; sm = sp'
Htwosite = Float64[sz[s1,s1p]*sz[s2,s2p]+0.5*(sp[s1,s1p]*sm[s2,s2p]+sm[s1,s1p]*sp[s2,s2p])
          for s1 = 1:2, s2 = 1:2, s1p = 1:2, s2p = 1:2]
#tau = 0.01
tau = 0.01im
taugate = reshape(expm(-tau*reshape(Htwosite,4,4)),2,2,2,2) # s1,s1p,s2,s2p?

B = cell(n)
n = 16
A = [zeros(1,2,1) for i = 1:n]
for i = 1:n
  A[i][1,iseven(i)? 2:1,1] = complex(1.0)
  #A[i] = complex(A[i])
  #B[i] = A[i]
end

#A[1][1,iseven(1)? 2:1,1] = complex(1.0)

wf  = MPS(A,1)
moveto!(wf,n)
normsq3(wf)


energy_oc(wf)
mp = sweep(wf,10)
energy_contra(mp)
energy_oc(mp)

energy_oc(moveto!(wf,n))
energy_oc(moveto!(wf,1))

normsq3(wf)                         #normalized?
(wf,tstep,E) = TEBD(wf,100,10)      #do TEBD
energy = energy_contra(wf)

Eexact = fill(-8.68247,100)         #exact energy   plot
#plot(tstep, E, tstep, Eexact, color="red", linewidth=2.0, linestyle="-")
fig, ax = subplots()
ax[:plot](tstep, E, linewidth=2, alpha=0.6, label="time evolution E")
ax[:plot](tstep, Eexact, linewidth=2, alpha=0.6, linestyle = "--", label="exact E")
title("size = 20, Beta = 1, tau = 0.01")
xlabel("tstep")
ylabel("energy")
ax[:legend]()


function TEBD(psi::MPS,nt,m)
  l = size(psi.A)[1]
  tstep = zeros(nt)
  E = zeros(nt)
  for t = 1:nt
    psi = sweep(psi,m)
    newpsi = tensor_reverse(psi)
    newpsi = sweep(newpsi,m)
    tstep[t] = t
    E[t] = energy_contra(newpsi)
    psi = tensor_reverse(newpsi)
  end
  return (psi,tstep,E)
end

function tensor_reverse(psi::MPS)
  l = size(psi.A)[1]
  newpsiA = 0*psi.A
  for i = l:-1:1
    psiAi = psi.A[i]
    (a,b,c) = size(psiAi)
    tensor_i = zeros(c,b,a)
    @tensor tensor_i[c,b,a] = psiAi[a,b,c]
    newpsiA[l-i+1] = tensor_i
  end
  newoc = l-psi.oc+1
  return MPS(newpsiA,newoc)
end

function sweep(psi::MPS,m)
  l = size(psi.A)[1]
  B = cell(l)
    for i = 1:l-1
      Ai = psi.A[i]
      Ai1 = psi.A[i+1]
      @tensor begin
        AA[a,f,g,e] := Ai[a,b,c]*Ai1[c,d,e]*taugate[b,d,f,g]
        #AA[a,b,d,e] := Ai[a,b,c]*Ai1[c,d,e]
      end
      #(psi.A[i],psi.A[i+1])=dosvdtoright(AA,m)
      (B[i],B[i+1]) = dosvdtoright(AA,m)
    end
    psi.oc = l
    #return psi
    return B
end

psi = wf
Ai = psi.A[1]
Ai1 = psi.A[1+1]
@tensor AA[a,f,g,e] := Ai[a,b,c]*Ai1[c,d,e]*taugate[b,d,f,g]
(a1,a2)=dosvdtoright(AA,10)
(psi.A[1],psi.A[1+1])=dosvdtoright(AA,10)
 B[1] = a1


psinew = sweep(wf,10)
mp1 = MPS(psinew,1)

sweep(mp1,10)
(mp1,tstep,E) = TEBD(mp1,100,10)      #do TEBD


function dosvdtoright(AA,m::Int64)
  (a,b,c,d) = size(AA)
  AA = reshape(AA,a*b,c*d)
  mpsl = svd(AA)[1]
  spectrum = svd(AA)[2]
  right = svd(AA)[3]
  D = zeros(m,m)
  Dm = length(spectrum)
  if Dm >m
    [D[i,i] = spectrum[i] for i = 1:m]
    right = D*transpose(right[:,1:m])
    return (reshape(mpsl[:,1:m],a,b,m), reshape(right,m,c,d))
  else
    [D[i,i] = spectrum[i] for i = 1:Dm]
    right = D[1:Dm,1:Dm]*transpose(right[:,1:Dm])
    return (reshape(mpsl[:,1:Dm],a,b,Dm), reshape(right,Dm,c,d))
  end
end

function dosvdtoleft(AA,m::Int64)
  (a,f,g,e) = size(AA)
  AA = reshape(AA,a*f,g*e)
  mpsr = svd(AA)[1]
  spectrum = svd(AA)[2]
  left = svd(AA)[3]
  D = zeros(m,m)
  Dm = length(spectrum)
  if Dm >m
    [D[i,i] = spectrum[i] for i = 1:m]
    left = D*transpose(left[:,1:m])
    return (reshape(left,a,b,m),reshape(mpsr[:,1:m],m,c,d))
  else
    [D[i,i] = spectrum[i] for i = 1:Dm]
    left = D[1:Dm,1:Dm]*transpose(left[:,1:Dm])
    return (reshape(left,e,g,Dm),reshape(mpsr[:,1:Dm],Dm,f,a))
  end
end
