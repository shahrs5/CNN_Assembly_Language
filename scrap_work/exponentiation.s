#define STDOUT 0xd0580000

.text
.global _start

_start:
    li   x3, 0xd0580000   # VeeR’s tohost address
    la a0 ,dense
    la a1 ,probabilities
    
    call softmax

    j  _finish

 softmax:
    mv s0 ,a0 #input dense
    mv s1 ,a1 #output probabilities
    mv s2 ,a1 #output probabilities second pass

    li t0,8  
    vsetvli t0,t0,e32 
    vmv.v.i v4, 0 #intialize accumulator

    li t0, 10  # total values
    exponentiation:
        vsetvli t1,t0,e32  # set vector load
        sub t0,t0,t1     # update remaining values
        vle32.v v0,(s0)   #load values from input

        addi t3, zero, 1                # t0 = 1
        fcvt.s.w f0, t3                 # f0 = float(1)
        vfmv.v.f v1, f0                 # v1 = broadcast 1.0
        vfmv.v.f v2, f0                 # v2 = term vector


        li t2, 100              # Load constant 1
        li t3,1

        exp_loop:
            bge t3 , t2 ,exp_done

            vfmul.vv v2 ,v2 ,v0 # =x/i(prev)*x

            fcvt.s.w f0,t3 # load i
            vfmv.v.f v3,f0 # populate v3 with i

            vfdiv.vv v2,v2,v3 # = x/i(prev) *x/i
            vfadd.vv v1 ,v1,v2 #accumulate 

            add t3 ,t3,1 # incement pointer

            j exp_loop
        exp_done:

            vse32.v v1,(s1) # store at probabilities 
            
            slli t6,t1,2   # ofset input and probability
            add s1 ,s1,t6
            add s0, s0 ,t6

            vfredosum.vs v4,v1, v4  # accumalate exp


            bnez t0 , exponentiation

    vfmv.f.s f0 ,v4 # move sum to f0
    li t0, 10
    secondpass:

     vsetvli t1,t0,e32  # set vector load
     vfmv.v.f v4,f0 # populate vector with sum
     sub t0,t0,t1     # update remaining values
     slli t1,t1,2   # input and probability

     vle32.v v0,(s2)   #load values from second pass

     vfdiv.vv v0,v0,v4 # divide each stored value with accumulated value
    
     vse32.v v0, 0(s2) #store back
     add s2 ,s2,t1 # update output pointer
     bnez t0, secondpass

     ret

_finish:
    li x3, 0xd0580000
    addi x5, x0, 0x1              # Signal 1 to end program
    sb x5, 0(x3)
    beq x0, x0, _finish
.rept 100
    nop
.endr




.data
dense: .float -5.791317, -7.479757, -11.160609, -9.319376, -9.348123, 3.908951, 8.004395, -22.608963, -3.778416, -10.471082
probabilities: .space 40

