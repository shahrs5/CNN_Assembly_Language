.data
dense:
probabilities:







.text


    _start
    la a0 dense
    la a1 probabilities
    la a2 probabilities
    call softmax





    _finish:
        li   x3, 0xd0580000   # VeeR’s tohost address
        addi x5, x0, 0xff     # status ≔ 0xff (often “success”)
        sb   x5, 0(x3)        # store byte
        # ebreak
    beq  x0, x0, _finish  # spin forever



 softmax:
    mv s0 ,a0 #input dense
    mv s1 ,a1 #output probabilities
    mv s2 ,a1 #output probabilities second pass
    li t0,10   # total values

    vsetvli zero,t0,e32 
    vmv.v.i v4, 0 #intialize accumulator


    exponentiation:
        li t4, 1               # Load constant 1
        li t3,1

        exp_loop:
            bge t3 , t2 ,exp_done


            vfmul.vv v2 ,v2 ,v0 # =x/i(prev)*x

        li t3,1

        exp_loop:
            bge t3 , t2 ,exp_done


            vfmul.vv v2 ,v2 ,v0 # =x/i(prev)*x

            fcvt.s.w f0,t3 # load i
            vfmv.v.f v3,f0 # populate v3 with i

            vfdiv.vv, v2,v2,v3 # = x/i(prev) *x/i
            vfadd.vv v1 ,v1,v2 #accumulate 

            add t3 ,t3,1 # incement pointer

            j exp_loop
        exp_done:

            vse32.v v1,(s1) # store at probabilities 
            muli t6,t1,4   # ofset input and probability
            add s1 ,s1,t6
            add s0, s0 ,t6

            vfredosum.vs v4,v4  # accumalate exp


            benz t1 , exponentiation

    vfmv.f.s f0 ,v0 # move sum to f0
    li t0 10
    secondpass:

     vsetvli t1,t0,e32  # set vector load

    vfmv.v.f v4,f0 # populate vector with sum
     sub t0,t0,t1     # update remaining values
     vle32.v v0,(s2)   #load values from second pass

     vfdiv.vv v0,v0,v4 # 
    
    vse32.v v0(s2) #store back
     muli t6,t1,4   # ofset input and probability
     add s2,s2,t6   
         fcvt.s.w f0,t3 # load i
            vfmv.v.f v3,f0 # populate v3 with i

            vfdiv.vv, v2,v2,v3 # = x/i(prev) *x/i
            vfadd.vv v1 ,v1,v2 #accumulate 

            add t3 ,t3,1 # incement pointer

            j exp_loop
        exp_done:

            vse32.v v1,(s1) # store at probabilities 
            muli t6,t1,4   # ofset input and probability
            add s1 ,s1,t6
            add s0, s0 ,t6

            vfredosum.vs v4,v4  # accumalate exp


            benz t1 , exponentiation

    vfmv.f.s f0 ,v0 # move sum to f0
    li t0 10
    secondpass:

     vsetvli t1,t0,e32  # set vector load

    vfmv.v.f v4,f0 # populate vector with sum
     sub t0,t0,t1     # update remaining values
     vle32.v v0,(s2)   #load values from second pass

     vfdiv.vv v0,v0,v4 # 
    
    vse32.v v0(s2) #store back
     muli t6,t1,4   # ofset input and probability
     add s2,s2,t6   
     fcvt.s.w f0, t4   # convert 1 to float
        vfmv.v.f v1,f0  # populate vector 1 to 3 with 1
        vfmv.v.f v2,f0

        vsetvli t1,t0,e32  # set vector load

        vle32.v v0,(s0)   #load values from dense 
        sub t0,t0,t1     # update remaining values


        li t2,1000
        li t3,1

        exp_loop:
            bge t3 , t2 ,exp_done


            vfmul.vv v2 ,v2 ,v0 # =x/i(prev)*x

            fcvt.s.w f0,t3 # load i
            vfmv.v.f v3,f0 # populate v3 with i

            vfdiv.vv, v2,v2,v3 # = x/i(prev) *x/i
            vfadd.vv v1 ,v1,v2 #accumulate 

            add t3 ,t3,1 # incement pointer

            j exp_loop
        exp_done:

            vse32.v v1,(s1) # store at probabilities 
            ssli t6,t1,2   # ofset input and probability
            add s1 ,s1,t6
            add s0, s0 ,t6

            vfredosum.vs v4,v4  # accumalate exp


            bnez t1 , exponentiation

    vfmv.f.s f0 ,v0 # move sum to f0
    li t0 10
    secondpass:

     vsetvli t1,t0,e32  # set vector load

    vfmv.v.f v4,f0 # populate vector with sum
     sub t0,t0,t1     # update remaining values
     vle32.v v0,(s2)   #load values from second pass

     vfdiv.vv v0,v0,v4 # 
    
    vse32.v v0(s2) #store back
     muli t6,t1,4   # ofset input and probability
     add s2,s2,t6 

     bnez t1 ,secondpass  

ret



  