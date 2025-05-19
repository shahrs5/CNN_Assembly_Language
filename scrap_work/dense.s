
.data
flattened_pool:
weight_matrix:
bias_vector:
final_output:.space 40







.text


    _start
    la   a0, flattened_pool   # Load address of pooled feature maps
    la   a1, weight_matrix # Load address of fully connected layer weights
    la   a2, bias_vector   # Load address of fully connected layer biases
    la   a3, final_output  # Load address where final output will be stored

    call denselayer        # Call dense layer computation for this neuron

    




    _finish:
        li   x3, 0xd0580000   # VeeR’s tohost address
        addi x5, x0, 0xff     # status ≔ 0xff (often “success”)
        sb   x5, 0(x3)        # store byte
        # ebreak
    beq  x0, x0, _finish  # spin forever










    .globl denselayer       # Make denselayer function globally accessible
    denselayer:             # Start of dense layer function
    mv s0 ,a0               #flatten pool
    mv s1 ,a1               #weight matrix
    mv s2 ,a2               #bias vector
    mv s3 ,a3               #final output
    li t2,10
    li t4, 40

    li t0,0 
    dense_outer:
        li a7,8
        vsetvli a7,a7,e32
        vmv.v.i v3,0 
        mv s0,a0
        li t5 ,1152 
        li t1,0
        slli a7,t0,2
        add s4,s1,a7
        dense_inner:

            vsetvli t6,t5, e32
            sub t5, t5,t6
            slli a7,t6,2
            
            vle32.v v0 , (s0)
            add s0,s0,a7


            vlse32.v v1 , (s4),t4
            mul a7,t6,t4
            add s4 ,s4,a7

            vfmul.vv v2,v1,v0

            vfredosum.vs v3,v2,v3

            bnez t5 , dense_inner


        done_inner:

        flw f0,(s2)
        vfmv.f.s f1,v3
        fadd f1,f1,f0
        fsw f1,(s3)
    
        addi s3,s3,4
        addi s2,s2,4

    addi t0 ,t0 ,1
    blt t0,t2,dense_outer
    done_outer:                                                                                                                                                                                        


    ret
    