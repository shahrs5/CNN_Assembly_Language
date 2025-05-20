#define STDOUT 0xd0580000



.section .text              # Indicates this is executable code section
.globl _start             # Defines _start as a global symbol accessible outside this file

_start:                   # Main entry point of the program
    # Registers:
    # a0: input_matrix    # Will hold pointer to input image (likely 28x28)
    # a1: filter_kernel   # Will hold pointer to filter weights (5x5 kernels)
    # a2: output_filter   # Will hold pointer to where feature maps will be stored
    # a3: filter_bias     # Will hold pointer to filter bias values
    la   a0, input_matrix # Load address of input_matrix into a0
    la   a1, filter_kernel # Load address of filter kernels into a1
    la   a2, output_filter # Load address of output feature maps into a2
    la   a3, filter_bias   # Load address of filter biases into a3

    li   t0, 0             # Initialize filter index f = 0
    conv_loop:                 # Start of convolution loop over 8 filters
        
        call conv2d            # Call convolution function for this filter
        li   t1, 25            # Each filter has 25 elements (5x5 kernel)
        slli t1, t1, 2         # t2 = f * 25 * 4 (bytes) - shift left logical immediate by 2 (multiply by 4)
        add  a1, a1, t1        # a4 = address of current filter's kernel (filter_kernel + offset)

        li   t1, 576           # Each output feature map has 576 elements (24x24)
        slli t1, t1, 2         # t4 = f * 576 * 4 (bytes) - multiply by 4 for float size
        add  a2, a2, t1            

        li t1 ,1
        slli t1 ,t1, 2         # t6 = f * 4 (bytes) - offset for current filter's bias
        add  a3, a3, t1       # a6 = address of current filter's bias value

        addi t0, t0, 1         # Increment filter index f++
        li   t5, 8             # We have 8 filters total
    blt  t0, t5, conv_loop # If f < 8, loop back to process next filter

  
    # Maxpool each of the 8 output feature maps
    la   a0, output_filter # Load address of convolution output
    la   a1, output_pool   # Load address where pooled results will be stored

    li   t0, 0             # Initialize filter index f = 0
    maxpool_loop:              # Start of maxpooling loop over 8 feature maps
        call maxpool           # Call maxpooling function for this feature map
        li   t1, 576           # Each feature map has 576 elements (24x24)
        slli t1, t1, 2         # t1 =  576 * 4 (bytes)
        add  a0, a0, t1        # a0 = address of current feature map

        li   t1, 144           # Each pooled output has 144 elements (12x12)
        slli t1, t1, 2         # t4 = 144 * 4 (bytes)
        add  a1, a1, t1        # a1 = address where pooled output will be stored

    addi t0, t0, 1         # Increment filter index f++
    li   t5, 8             # We have 8 feature maps to pool
    blt  t0, t5, maxpool_loop # If f < 8, loop back to process next feature map

    la a0 ,output_pool
    la a1 ,flattened_pool

    call flatten


    la   a0, flattened_pool   # Load address of pooled feature maps
    la   a1, weight_matrix # Load address of fully connected layer weights
    la   a2, bias_vector   # Load address of fully connected layer biases
    la   a3, final_output  # Load address where final output will be stored


    call denselayer
    

    la a0, final_output    # Load address of neural network raw outputs
    la a1, probability_matrix # Load address where softmax probabilities will be stored
  
    call softmax           # Apply softmax to convert raw outputs to probabilities

    
    call print_input
    call print_output_filter
    call print_output_pool
    call print_flatten
    call print_dense
    call print_probabilities




    _finish:
        li   x3, 0xd0580000   # VeeR’s tohost address
        addi x5, x0, 0xff     # status ≔ 0xff (often “success”)
        sb   x5, 0(x3)        # store byte
        # ebreak
    beq  x0, x0, _finish  # spin forever

#----------------------------------------------------------------------
# conv2d: Convolution 5x5 on 28x28 → 24x24 output
# Inputs:
#   a0 = input base ptr
#   a1 = filter base ptr (25 floats)
#   a2 = output base ptr
#   a3 = bias ptr (float *)
# Registers used:
#   s0 = input ptr
#   s1 = not used
#   s2 = output ptr
#   s3 = filter ptr
#   s4 = temp filter row ptr
#   t0-t6 = loop counters, offsets
#   v0-v4 = vector registers for computation

.globl conv2d              # Make conv2d function globally accessible
conv2d:                    # Start of convolution function
    addi sp, sp, -28       # Allocate space on stack for t0–t6 (7 regs × 4 bytes)
    sw   t0, 0(sp)
    sw   t1, 4(sp)
    sw   t2, 8(sp)
    sw   t3, 12(sp)
    sw   t4, 16(sp)
    sw   t5, 20(sp)
    sw   t6, 24(sp)
    li   t6, 5             # Kernel size is 5x5
    mv   s0, a0            # s0 = input matrix address
    mv   s2, a2            # s2 = output matrix address
    mv   s3, a1            # s3 = filter kernel address
    flw  f1, 0(a3)         # Load filter bias into f1

    li   t0, 0             # Initialize output row counter i = 0
    .conv_i:                   # Start of loop over output rows
        li   t1, 0             # Initialize output column counter j = 0
        li   t6, 24            # Output size is 24x24
        .conv_j:                   # Start of loop over output columns
            li   t4, 28            # Input width is 28
            mul  t5, t0, t4        # t5 = i * 28 (row offset in input)
            add  t5, t5, t1        # t5 = i * 28 + j (position in input)
            slli t5, t5, 2         # t5 = (i * 28 + j) * 4 (byte offset)
            add  t3, s0, t5        # t3 = address of input[i][j]

            mv   s4, s3            # s4 = filter kernel base address
            li   t2, 0             # Initialize filter row counter fi = 0
            li   t4, 5             # Filter size is 5x5 (replaced t7 with t4)
            vmv.v.i v4, 0          # Initialize accumulator vector to 0
            .conv_fi:                  # Start of loop over filter rows
            
                vsetvli t5, t4, e32    # Set vector length for 32-bit elements (replaced t7 with t4)
                vle32.v v1, (s4)       # Load filter row into v1 vector register
                vle32.v v0, (t3)       # Load input window row into v0 vector register
                vfmul.vv v2, v0, v1    # Multiply input and filter element-wise
                vfredosum.vs v4, v2, v4 # Accumulate sum of products into v4
                addi s4, s4, 20        # Move to next filter row (5 floats * 4 bytes)
                addi t3, t3, 112       # Move to next input row (28 floats * 4 bytes)

            addi t2, t2, 1         # Increment filter row counter
            blt  t2, t4, .conv_fi  # If not done with filter rows, continue (replaced t7 with t4)

            vfmv.f.s f0, v4        # Move accumulated sum from vector to scalar
            fadd.s   f0, f0, f1    # Add bias to convolution result

            li   t6, 24            # Output width is 24
            mul  t5, t0, t6        # t5 = i * 24 (row offset in output)
            add  t5, t5, t1        # t5 = i * 24 + j (position in output)
            slli t5, t5, 2         # t5 = (i * 24 + j) * 4 (byte offset)
            add  t5, s2, t5        # t5 = address of output[i][j]
            fsw  f0, 0(t5)         # Store result in output matrix

        addi t1, t1, 1         # Increment column counter j++
        blt  t1, t6, .conv_j   # If j < 24, continue with next column

    addi t0, t0, 1         # Increment row counter i++
    blt  t0, t6, .conv_i   # If i < 24, continue with next row


    lw   t0, 0(sp)
    lw   t1, 4(sp)
    lw   t2, 8(sp)
    lw   t3, 12(sp)
    lw   t4, 16(sp)
    lw   t5, 20(sp)
    lw   t6, 24(sp)
    addi sp, sp, 28 

    ret                    # Return from function

#----------------------------------------------------------------------
# maxpool: 2x2 max pool on 24x24 input to 12x12 output
# Inputs:
#   a0 = input base ptr (24x24)
#   a1 = output base ptr (12x12)
# Registers:
#   s0 = input ptr
#   s1 = output ptr
#   t0 = row i
#   t1 = col j
#   v0, v1 = rows
#   v2 = max
#   v4 = reduced max

.globl maxpool             # Make maxpool function globally accessible
maxpool:                   # Start of maxpooling function
    addi sp, sp, -28       # Allocate space on stack for t0–t6 (7 regs × 4 bytes)
    sw   t0, 0(sp)
    sw   t1, 4(sp)
    sw   t2, 8(sp)
    sw   t3, 12(sp)
    sw   t4, 16(sp)
    sw   t5, 20(sp)
    sw   t6, 24(sp)
    
    mv   s0, a0            # s0 = input feature map address
    mv   s1, a1            # s1 = output pooled map address
    li   t0, 0             # Initialize output row counter i = 0
    .pool_i:                   # Start of loop over output rows
        li   t1, 0             # Initialize output column counter j = 0
        .pool_j:                   # Start of loop over output columns
            slli t2, t0, 1         # t2 = i * 2 (input row)
            li   t3, 24            # Input width is 24
            mul  t2, t2, t3        # t2 = i * 2 * 24 (input row offset)
            slli t3, t1, 1         # t3 = j * 2 (input column)
            add  t2, t2, t3        # t2 = i * 2 * 24 + j * 2 (input position)
            slli t2, t2, 2         # t2 = (i * 2 * 24 + j * 2) * 4 (byte offset)
            add  t4, s0, t2        # t4 = address of input[i*2][j*2]
            addi t5, t4, 96        # t5 = address of input[i*2+1][j*2] (next row, 24*4=96)

            li t3, 2
            vsetvli t2, t3, e32     # Set vector length for 2 elements
            
            slli t3, t1, 1
            vle32.v v0, (t4)       # Load top 2x1 window into v0
            vle32.v v1, (t5)       # Load bottom 2x1 window into v1
            vmax.vv v2, v0, v1     # Find max between rows
            vsetvli t2, zero, e32  # Set vector length to 8 for broadcast
            vmv.v.i v4, 0          # Initialize reduction vector does RELU aswell

            # li t6 ,0xff800000    # negative infinity if you want to consider negative asweel
            # fmv.w.x  f5, t6      # Move bit pattern into float register
            # vfmv.v.f v4, f5      # Broadcast float to vector


            vfredmax.vs v4, v2, v4 # Find maximum value in window
            vfmv.f.s f6, v4        # Move max value from vector to scalar
            
            li   t6, 12            # Output size is 12x12
            mul  t3, t0, t6        # t3 = i * 12 (output row offset)
            add  t3, t3, t1        # t3 = i * 12 + j (output position)
            slli t3, t3, 2         # t3 = (i * 12 + j) * 4 (byte offset)
            add  t3, s1, t3        # t3 = address of output[i][j]
            fsw  f6, 0(t3)         # Store max value in output matrix

        addi t1, t1, 1         # Increment column counter j++
        blt  t1, t6, .pool_j   # If j < 12, continue with next column

    addi t0, t0, 1         # Increment row counter i++
    blt  t0, t6, .pool_i   # If i < 12, continue with next row



    lw   t0, 0(sp)
    lw   t1, 4(sp)
    lw   t2, 8(sp)
    lw   t3, 12(sp)
    lw   t4, 16(sp)
    lw   t5, 20(sp)
    lw   t6, 24(sp)
    addi sp, sp, 28 

    ret                    # Return from function


#----------------------------------------------------------------------
# flatten: Converts a 12x12 matrix (from maxpool) into a 1D vector
#          using vector strided load and store (RISC-V Vector)
#
# Inputs:
#   a0 = base pointer to 12x12 maxpool output (576 elements, float32)
#   a1 = base pointer to flatten output (1D array of 576 float32)
#
# Registers:
#   s0 = input pointer (maxpool matrix)
#   s1 = output pointer (flattened vector)
#   s2 = stride offset for strided load (576 bytes)
#   t0 = outer loop counter (row index, 0 to 11)
#   t1 = inner loop counter (column index, 0 to 11)
#   t2 = loop bound = 12 (dimension of the matrix)
#   t3 = number of vector elements to load/store per iteration (8 floats)
#   t4 = temporary register for vector length config
#   v0 = vector register for 8 loaded float32 values

.global flatten
flatten:

    mv s0 ,a0 # maxpool address
    mv s1 ,a1 # flatten address
    li t2 , 12 # dimension of matrix
    li t0 ,0 # row loop counter
    li t3 ,8 # vectors to load
    li s2 , 576 # offset

    row_loop:


        li t1,0 # column loop counter
        column_loop:
            
            vsetvli t4, t3, e32 #
            vlse32.v v0,(s0),s2 # strided segment load 576
            vse32.v v0,(s1) # save 8 values at flatten

            addi s0,s0,4 #offset maxpool
            addi s1 ,s1,32 # offset flatten

        addi t1 ,t1,1 #increment column counter
        blt t1,t2,column_loop
        end_column:

    addi t0 ,t0,1 # increment row counter
    blt t0,t2,row_loop
    end_row:

ret



#----------------------------------------------------------------------
# denselayer: Fully connected layer (dense layer) implementation
#             Performs: output[i] = dot(flatten, weights[i]) + bias[i]
#
# Inputs:
#   a0 = base pointer to input vector (flattened 12x12 = 144 float32)
#   a1 = base pointer to weight matrix (10x144, row-major, untransposed)
#   a2 = base pointer to bias vector (10 float32 values)
#   a3 = base pointer to output vector (10 float32 outputs)
#
# Registers:
#   s0 = pointer to flattened input vector (reused/reset per neuron)
#   s1 = pointer to weight matrix (start of all weights)
#   s2 = pointer to bias vector
#   s3 = pointer to output vector
#   s4 = working pointer for weights[i][j]
#   t0 = outer loop counter (for neurons 0 to 9)
#   t2 = constant 10 (number of output neurons)
#   t4 = constant 40 (stride in bytes = 10*4; used for strided weight loads)
#   t5 = number of elements left to process in inner loop (starts at 144)
#   t6 = vector length for each iteration of inner loop
#   v0 = vector register for input vector (flattened image)
#   v1 = vector register for weights[i][j]
#   v2 = vector product of v0 and v1
#   v3 = accumulator for dot product



.globl denselayer       # Make denselayer function globally accessible
    denselayer:             # Start of dense layer function
    mv s0 ,a0               #flatten pool
    mv s1 ,a1               #weight matrix
    mv s2 ,a2               #bias vector
    mv s3 ,a3               #final output
    li t2,10                #load constant 10 for dense outer loop
    li t4, 40               #load offset for weights ( weightd are default tensor flow format not updated )

    li t0,0                 #initialize variable for outer loop
    dense_outer:

        vsetvli t3,zero,e32  #set vector length to 8 for broadcast
        vmv.v.i v3,0         #set up an accumulator
        mv s0,a0             #reset flatten pool pointer
        li t5 ,1152          # load number of elements to process
        li t1,0              # dead code ???
        slli t3,t0,2         # calculate offset for weight matriz
        add s4,s1,t3         #working pointer for weight matrix offest so weight [i][0]
        dense_inner:        

            vsetvli t6,t5, e32    #set vectot length to min(8,remaining elements)
            sub t5, t5,t6         # update remaining elements
            slli t3,t6,2          # caculate offset
            
            vle32.v v0 , (s0)    #load flatten
            add s0,s0,t3         # offset flatten


            vlse32.v v1 , (s4),t4 # strided segement load from weight matrix (as its not transposed)
            mul t3,t6,t4          #caculate next offset by value loaded *40
            add s4 ,s4,t3         #offset s4

            vfmul.vv v2,v1,v0     #multiply both vectors

            vfredosum.vs v3,v2,v3 #vector reduce sum into accumulator

            bnez t5 , dense_inner #loop untill values exhuasted


        done_inner:

        flw f0,(s2)      #load bias value
        vfmv.f.s f1,v3   #load move accumlator value
        fadd.s f1,f1,f0  #add bias to accumlator value
        fsw f1,(s3)      #store at location
    
        addi s3,s3,4 #offset pointer for storage
        addi s2,s2,4 #offset pointer for bias

    addi t0 ,t0 ,1    #increment variable for outer loop
    blt t0,t2,dense_outer #loop untill 10
    done_outer:                                                                                                                                                                                        


    ret
    



#----------------------------------------------------------------------
# softmax: Computes softmax over a 10-element vector using Taylor
#          series approximation for exponentiation.
#
# Inputs:
#   a0 = base pointer to input vector (10 float32 values from dense layer)
#   a1 = base pointer to output vector (softmax probabilities)
#
# Registers:
#   s0 = input pointer (dense layer output)
#   s1 = output pointer for first pass (exp(x))
#   s2 = output pointer for second pass (normalization)
#   t0 = loop counter / remaining values to process
#   t1 = vector length per iteration
#   t2 = constant upper limit for Taylor expansion (100 terms)
#   t3 = Taylor loop counter
#   t4 = temporary for vector config
#   t6 = byte offset for pointer updates
#   f0 = scalar float temp (loop index or sum)
#   v0 = input vector chunk
#   v1 = accumulator for exp(x) approximation
#   v2 = term in Taylor expansion
#   v3 = vector loop index (float)
#   v4 = accumulator for total sum of exponentials (used for normalization)


.globl softmax

softmax:
    mv s0 ,a0 #input dense
    mv s1 ,a1 #output probabilities
    mv s2 ,a1 #output probabilities second pass

    vsetvli t4,zero,e32 #set vector length to 8 for broadcast
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
            # bge t3 , t2 ,exp_done

            vfmul.vv v2 ,v2 ,v0 # =x/i(prev)*x

            fcvt.s.w f0,t3 # load i
            vfmv.v.f v3,f0 # populate v3 with i

            vfdiv.vv v2,v2,v3 # = x/i(prev) *x/i
            vfadd.vv v1 ,v1,v2 #accumulate 

            add t3 ,t3,1 # incement pointer

            blt t3, t2 ,exp_loop
            # j exp_loop
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





#----------------------------------------------------------------------
.globl print_input              # Make print function globally accessible

print_input:                     # Start of print function (debug output)
    la s0, input_matrix
    li s1 ,784 # 28x28
.print_loop_input:               # Start of print loop
    beq  s1, zero, .print_done_input # If no elements left, finish
    vsetvli t1, s1, e32    # Set vector length for remaining elements
    li t0, 1               # Debug instruction (no actual operation)
    li t0, 2               # Debug instruction (no actual operation)
    li t0, 3               # Debug instruction (no actual operation)

    vle32.v v0, (s0)       # Load vector chunk into v0 (not printed, just loaded)


    slli t2, t1, 2         # t2 = processed elements * 4 (bytes)
    add  s0, s0, t2       # Advance vector pointer
    sub  s1, s1, t1       # Reduce remaining element count
    j    .print_loop_input      # Continue loop
.print_done_input:               # End of print function
    ret                    # Return from function



.globl print_output_filter             # Make print function globally accessible

print_output_filter:                     # Start of print function (debug output)
    la s0, output_filter
    li s1 ,4608 #24x24x8
.print_loop_filter:               # Start of print loop
    beq  s1, zero, .print_done_filter # If no elements left, finish
    vsetvli t1, s1, e32    # Set vector length for remaining elements
    li t0, 2              # Debug instruction (no actual operation)
    li t0, 3               # Debug instruction (no actual operation)
    li t0, 4               # Debug instruction (no actual operation)

    vle32.v v0, (s0)       # Load vector chunk into v0 (not printed, just loaded)

    slli t2, t1, 2         # t2 = processed elements * 4 (bytes)
    add  s0, s0, t2       # Advance vector pointer
    sub  s1, s1, t1       # Reduce remaining element count
    j    .print_loop_filter       # Continue loop
.print_done_filter:               # End of print function
    ret                    # Return from function






.globl print_output_pool          # Make print function globally accessible

print_output_pool:                 # Start of print function (debug output)
    la s0, output_pool
    li s1 ,1152 
.print_loop_pool:               # Start of print loop
    beq  s1, zero, .print_done_pool # If no elements left, finish
    vsetvli t1, s1, e32        # Set vector length for remaining elements
    li t0, 3                   # Debug instruction (no actual operation)
    li t0, 4                   # Debug instruction (no actual operation)
    li t0, 5                   # Debug instruction (no actual operation)

    vle32.v v0, (s0)           # Load vector chunk into v0 (not printed, just loaded)

    slli t2, t1, 2             # t2 = processed elements * 4 (bytes)
    add  s0, s0, t2            # Advance vector pointer
    sub  s1, s1, t1            # Reduce remaining element count
    j    .print_loop_pool      # Continue loop
.print_done_pool:              # End of print function
    ret                        # Return from function



.globl print_flatten               # Make print function globally accessible

print_flatten:                     # Start of print function (debug output)
    la s0, flattened_pool
    li s1 ,1152 #12x12x8
.print_loop_flatten:                   # Start of print loop
    beq  s1, zero, .print_done_flatten # If no elements left, finish
    vsetvli t1, s1, e32    # Set vector length for remaining elements
    li t0, 4               # Debug instruction (no actual operation)
    li t0, 5               # Debug instruction (no actual operation)
    li t0, 6               # Debug instruction (no actual operation)

    vle32.v v0, (s0)       # Load vector chunk into v0 (not printed, just loaded)

    

    slli t2, t1, 2        # t2 = processed elements * 4 (bytes)
    add  s0, s0, t2       # Advance vector pointer
    sub  s1, s1, t1       # Reduce remaining element count
    j    .print_loop_flatten      # Continue loop
.print_done_flatten:              # End of print function
    ret                    # Return from function




.globl print_dense            # Make print function globally accessible

print_dense:                     # Start of print function (debug output)
    la s0, final_output
    li s1 ,10 #10
.print_loop_dense:                   # Start of print loop
    beq  s1, zero, .print_done_dense # If no elements left, finish
    vsetvli t1, s1, e32    # Set vector length for remaining elements
    li t0, 5               # Debug instruction (no actual operation)
    li t0, 6               # Debug instruction (no actual operation)
    li t0, 7               # Debug instruction (no actual operation)

    vle32.v v0, (s0)       # Load vector chunk into v0 (not printed, just loaded)

    

    slli t2, t1, 2             # t2 = processed elements * 4 (bytes)
    add  s0, s0, t2            # Advance vector pointer
    sub  s1, s1, t1            # Reduce remaining element count
    j    .print_loop_dense     # Continue loop
.print_done_dense:             # End of print function
    ret                        # Return from function



.globl print_probabilities             # Make print function globally accessible

print_probabilities:                     # Start of print function (debug output)
    la s0, probability_matrix
    li s1 ,10 #10
.print_loop_prob:               # Start of print loop
    beq  s1, zero, .print_done_prob# If no elements left, finish
    vsetvli t1, s1, e32    # Set vector length for remaining elements
    li t0, 6               # Debug instruction (no actual operation)
    li t0, 7               # Debug instruction (no actual operation)
    li t0, 8               # Debug instruction (no actual operation)

    vle32.v v0, (s0)       # Load vector chunk into v0 (not printed, just loaded)


    slli t2, t1, 2        # t2 = processed elements * 4 (bytes)
    add  s0, s0, t2       # Advance vector pointer
    sub  s1, s1, t1       # Reduce remaining element count
    j    .print_loop_prob # Continue loop
.print_done_prob:         # End of print function
    ret                   # Return from function


.section .data
.align 4


input_matrix:
.float 0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000
.float 0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000
.float 0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000
.float 0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000
.float 0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000
.float 0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000
.float 0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000
.float 0.000,0.000,0.000,0.000,0.000,0.000,0.329,0.725,0.624,0.592,0.235,0.141,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000
.float 0.000,0.000,0.000,0.000,0.000,0.000,0.871,0.996,0.996,0.996,0.996,0.945,0.776,0.776,0.776,0.776,0.776,0.776,0.776,0.776,0.667,0.204,0.000,0.000,0.000,0.000,0.000,0.000
.float 0.000,0.000,0.000,0.000,0.000,0.000,0.263,0.447,0.282,0.447,0.639,0.890,0.996,0.882,0.996,0.996,0.996,0.980,0.898,0.996,0.996,0.549,0.000,0.000,0.000,0.000,0.000,0.000
.float 0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.067,0.259,0.055,0.263,0.263,0.263,0.231,0.082,0.925,0.996,0.416,0.000,0.000,0.000,0.000,0.000,0.000
.float 0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.325,0.992,0.820,0.071,0.000,0.000,0.000,0.000,0.000,0.000
.float 0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.086,0.914,1.000,0.325,0.000,0.000,0.000,0.000,0.000,0.000,0.000
.float 0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.506,0.996,0.933,0.173,0.000,0.000,0.000,0.000,0.000,0.000,0.000
.float 0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.231,0.976,0.996,0.243,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000
.float 0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.522,0.996,0.733,0.020,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000
.float 0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.035,0.804,0.973,0.227,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000
.float 0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.494,0.996,0.714,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000
.float 0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.294,0.984,0.941,0.224,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000
.float 0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.075,0.867,0.996,0.651,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000
.float 0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.012,0.796,0.996,0.859,0.137,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000
.float 0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.149,0.996,0.996,0.302,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000
.float 0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.122,0.878,0.996,0.451,0.004,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000
.float 0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.522,0.996,0.996,0.204,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000
.float 0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.239,0.949,0.996,0.996,0.204,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000
.float 0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.475,0.996,0.996,0.859,0.157,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000
.float 0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.475,0.996,0.812,0.071,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000
.float 0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000

filter_kernel:      
.float -0.581912, -0.004107, 0.262724, 0.455814, -0.015057
.float 0.019984, 0.490956, 0.561276, 0.021874, -0.502644
.float 0.340455, 0.550426, -0.004595, -0.533139, -0.786208
.float 0.268008, 0.258691, -0.454976, -0.337063, -0.362381
.float 0.335046, -0.172678, -0.346649, -0.435975, -0.158836
.float 0.243651, 0.123896, -0.202078, -0.227226, -0.137311
.float 0.053332, 0.146312, 0.011387, 0.016027, 0.119655
.float 0.036402, 0.063896, 0.162261, 0.361339, -0.226303
.float -0.027464, -0.108144, 0.002567, 0.396431, 0.270360
.float 0.151043, 0.066532, -0.403678, -0.131380, 0.182330
.float -0.704177, -0.824182, -1.239597, -0.841037, -0.729635
.float -0.912624, -0.811920, -0.527901, -0.426248, -0.452107
.float -0.922194, -0.228362, 0.094180, 0.014453, -0.001725
.float -0.529561, 0.135211, 0.462474, 0.433857, 0.545282
.float 0.077638, 0.526868, 0.441724, 0.489472, 0.494138
.float -0.975883, -0.222055, 0.139011, 0.619273, 0.207563
.float -1.084636, -0.188388, 0.116340, 0.459774, 0.068332
.float -0.548894, -0.316574, 0.154199, 0.419673, 0.118501
.float -0.332960, -0.267086, 0.041670, 0.181641, 0.228478
.float -0.485350, -0.047923, 0.161516, 0.235072, 0.278659
.float -0.198275, -0.118308, -0.186803, -0.179359, -0.043761
.float -0.307096, 0.226955, 0.371379, 0.128756, 0.077212
.float 0.413754, 0.192896, 0.112449, 0.173986, 0.152216
.float 0.084656, -0.278355, -0.538080, -0.102567, 0.028882
.float 0.093607, 0.250828, 0.265590, 0.217743, 0.259163
.float 0.025814, 0.533002, 0.745715, 0.800645, 0.604509
.float -0.494648, -0.446248, -0.103960, 0.085778, 0.286187
.float -0.253287, -1.107013, -1.487154, -1.166972, -0.423916
.float 0.462314, -0.090064, -0.636340, -0.393062, -0.351414
.float 0.274775, 0.706314, 0.601848, 0.436756, 0.031815
.float 0.473004, 0.179765, -0.289376, -0.370865, 0.059390
.float -0.244290, -0.107724, -0.281314, 0.024289, 0.083030
.float -0.700936, -0.220954, 0.360511, 0.355795, 0.261670
.float -0.025650, 0.356979, 0.244073, 0.115337, 0.041575
.float 0.453729, 0.147794, -0.192686, -0.517529, -0.047058
.float 0.067598, -0.017237, -0.504901, -0.610099, -0.246365
.float 0.326682, 0.290307, -0.360458, -0.523399, -0.304977
.float 0.359524, 0.415965, -0.381088, -0.493526, -0.439265
.float 0.340188, 0.614115, 0.035128, -0.032836, -0.327628
.float -0.128007, 0.511645, 0.535439, 0.327660, 0.090803


filter_bias:
    .float  -0.017957, -0.492205, 0.130652, -0.174127, -0.437835, 0.011152, -0.254810, -0.169736

    
output_filter:
    .rept 4608            # 24×24×8
      .float 0.0
    .endr

output_pool:
    .rept 1152            # 12×12×8
      .float 0.0
    .endr

 flattened_pool:
 .rept 1152            # 12×12×8
       .float 0.0
    .endr


weight_matrix: 
.float -0.057135, 0.253652, 0.057309, -0.240312, -0.275638, 0.304827, -0.283109, 0.029929, -0.332590, 0.213356, -0.272279, 0.008764, -0.092158, 0.210386, 0.122946, -0.215309, 0.017546, -0.211495, -0.293838, -0.099011, 0.048542, 0.371898, 0.005997, -0.139244, -0.054748, -0.032076, 0.146874, 0.084067, -0.128195, -0.055870, -0.361369, 0.121009, 0.011335, 0.145653, -0.185998, -0.324096, 0.128656, -0.299675, -0.194825, -0.218054, -0.269893, 0.102775, 0.008582, -0.014283, 0.087751, -0.116329, -0.072014, -0.179007, -0.150669, -0.150315, 0.153539, 0.350031, 0.040697, -0.184291, -0.016967, -0.167931, 0.047272, -0.100383, -0.152770, -0.081101, -0.173107, -0.006490, -0.105048, 0.138660, -0.019819, -0.172150, 0.087657, 0.015770, -0.241314, -0.144884, -0.171380, 0.337134, 0.025051, 0.065776, -0.079652, -0.251828, 0.071578, -0.426478, -0.026321, -0.117006, -0.187265, 0.188764, -0.174557, -0.187886, -0.287800, 0.311960, -0.263742, -0.064072, -0.349646, 0.195167, -0.177075, 0.122395, -0.043635, 0.223553, -0.351293, -0.446470, 0.355158, -0.323642, 0.093465, -0.179950, -0.020755, 0.140169, 0.063595, 0.059002, -0.239639, -0.244660, 0.104640, -0.044785, -0.006462, -0.165601, -0.049146, -0.117090, 0.040811, 0.142870, -0.240151, -0.366265, 0.194354, -0.060493, 0.048384, -0.664802, 0.048704, -0.029612, -0.068168, 0.161132, -0.308969, -0.142066, 0.169005, 0.094324, -0.054179, -0.394179, 0.231697, 0.355336, -0.162230, -0.031111, -0.110644, -0.078026, 0.129996, 0.117453, -0.126447, -0.047011, -0.233467, -0.059255, -0.018620, 0.134428, -0.239000, -0.088010, 0.236362, -0.098357, 0.009523, -0.156386, -0.077212, 0.179767, -0.137803, 0.058261, -0.292053, -0.218318, 0.088642, -0.004831, -0.142311, -0.503834, -0.534129, -0.117699, 0.002993, 0.119412, -0.261252, 0.139682, 0.479845, -0.234739, -0.406775, -0.008094, -0.059327, 0.001723, -0.095705, 0.284207, -0.355978, -0.451611, 0.347056, -0.540178, -0.111077, -0.458328, 0.089486, -0.100904, 0.029969, -0.067002, -0.245845, -0.126331, 0.169519, -0.041998, 0.171232, -0.262363, 0.074412, -0.273119, 0.027076, 0.047547, -0.318648, -0.331903, 0.371833, 0.003992, 0.138609, -0.242880, -0.063067, -0.275178, -0.018179, 0.056116, -0.293933, -0.216669, 0.312622, 0.034209, 0.023382, -0.220929, 0.009813, 0.201465, -0.007314, -0.105051, -0.092007, -0.124316, 0.085710, 0.179944, -0.038729, -0.097685, -0.223356, -0.283064, 0.034443, 0.240197, -0.145585, -0.202745, 0.310233, -0.343960, -0.463711, -0.173401, -0.167191, -0.025638, 0.006850, 0.012935, -0.098692, -0.390139, 0.110434, -0.020652, 0.017119, -0.352277, -0.257627, 0.333493, -0.112408, 0.219733, -0.226012, -0.283415, 0.435556, -0.190930, -0.101705, -0.588958, -0.140571, -0.419141, 0.316575, 0.148086, -0.624358, -0.069922, 0.265458, -0.790993, 0.033119, -0.644987, -0.003905, -0.002404, 0.088589, 0.047386, -0.246379, -0.011542, 0.055997, -0.165069, -0.013826, 0.051277, -0.145938, -0.137387, 0.130448, 0.014504, -0.567745, 0.054418, 0.371006, -0.229710, -0.053261, 0.018224, -0.146376, -0.350087, 0.097722, -0.070330, -0.539338, 0.165846, 0.189319, 0.153064, -0.005890, 0.260142, 0.013736, 0.065509, -0.131729, -0.227215, 0.082349, -0.052323, 0.031915, 0.179811, 0.021346, -0.093328, -0.036587, -0.451381, 0.164889, 0.033661, -0.364737, -0.314535, 0.106039, -0.304021, 0.278273, -0.392340, -0.156514, 0.043736, -0.016509, 0.035307, -0.262723, -0.132604, 0.261346, -0.066801, 0.038582, -0.108540, -0.284106, 0.292795, -0.019809, -0.330938, -0.187348, -0.122476, 0.475623, -0.298154, -0.347290, -0.960173, -0.121395, -0.300022, 0.127894, 0.317451, -0.275038, -0.471871, 0.007297, -0.339900, 0.173985, -0.399679, -0.048837, 0.028494, -0.001344, 0.003955, -0.241472, -0.057456, -0.081607, -0.060644, 0.132539, 0.167745, -0.166879, 0.021675, 0.128206, -0.031692, -0.256445, -0.096866, 0.039087, -0.211027, -0.034541, 0.188340, -0.169113, -0.124304, 0.113788, -0.094780, -0.384841, 0.109806, 0.044435, 0.070088, 0.020668, 0.159133, -0.057432, 0.027050, -0.021871, -0.081681, 0.042878, 0.060890, -0.100374, 0.083492, 0.059049, 0.021786, -0.092808, -0.287821, 0.080229, 0.119736, -0.246710, -0.046696, 0.140051, -0.408113, 0.174885, -0.460670, -0.085908, -0.016077, 0.106257, 0.022315, -0.291309, -0.093639, 0.000626, -0.193309, -0.045114, 0.080755, -0.090966, 0.175522, 0.144340, 0.005120, -0.240836, -0.319899, 0.528019, -0.687391, -0.071493, -0.683941, -0.054056, 0.027549, -0.040325, 0.229891, -0.340213, -0.179084, 0.194065, -0.616565, -0.015176, 0.037574, 0.046859, 0.088571, 0.001506, 0.107145, 0.000591, -0.001329, -0.049731, -0.007657, 0.018909, 0.093742, 0.100861, 0.028676, -0.105567, -0.021414, -0.075048, -0.156389, 0.085784, 0.037236, 0.110879, -0.122552, 0.011395, -0.010326, 0.154554, 0.057577, -0.497751, 0.037571, 0.231689, 0.133385, -0.082835, -0.003216, 0.082987, -0.045062, -0.062779, 0.100395, -0.080537, -0.013635, -0.150498, 0.085282, -0.115261, 0.032639, -0.124109, -0.223665, 0.083522, 0.215694, -0.225581, 0.017057, 0.109241, -0.285388, -0.143160, 0.096419, -0.010720, -0.146082, 0.117902, 0.093404, -0.213379, -0.229436, -0.064398, -0.149159, 0.001768, 0.115352, -0.051733, 0.570533, -0.287391, -0.159323, -0.069700, -0.629728, 0.371317, -0.458129, 0.093790, -0.264065, -0.518106, -0.187993, 0.336702, 0.301949, -0.577857, 0.013297, -0.110827, -0.750189, -0.031675, -0.195697, 0.097363, 0.082859, -0.096672, -0.018957, -0.146684, 0.049762, 0.041929, 0.104448, 0.069740, -0.044428, -0.056708, 0.019255, -0.079630, 0.018896, -0.147660, 0.064761, 0.062175, -0.163454, 0.087960, 0.015916, -0.023245, 0.098735, 0.014475, 0.141801, -0.270928, -0.034137, 0.107363, 0.104547, -0.152863, 0.113233, 0.029810, 0.022291, 0.008510, 0.021511, 0.164601, -0.057379, 0.017811, 0.108811, -0.116496, -0.084914, -0.161816, -0.178241, 0.130176, 0.012199, -0.096460, 0.194694, -0.145029, 0.025961, 0.144763, -0.058725, -0.035583, 0.139277, 0.039846, 0.147073, -0.002842, -0.147060, 0.137715, 0.097322, -0.061597, -0.118002, 0.058688, 0.256320, -0.353525, -0.068216, 0.071658, -0.057849, 0.399314, -0.351876, 0.096480, -0.476422, -0.291228, 0.003674, 0.187290, 0.147874, -0.399330, -0.021560, 0.116229, -0.666499, -0.299964, -0.177235, 0.112118, 0.086932, -0.133802, -0.115115, 0.042613, 0.089907, -0.042944, 0.056326, -0.077221, -0.106602, 0.015836, 0.172082, -0.162581, -0.147361, -0.343529, 0.205005, 0.192986, -0.061344, 0.013323, 0.065949, -0.373864, -0.004191, -0.069620, 0.166919, -0.513119, 0.067237, 0.345342, 0.209873, -0.254389, 0.061572, 0.004176, 0.046704, 0.016958, -0.030251, 0.116260, -0.083070, 0.044009, 0.178304, -0.058967, -0.084016, -0.088199, 0.061255, -0.099816, 0.114435, -0.102181, 0.289284, -0.088568, -0.370017, 0.018463, -0.221842, -0.146042, -0.010102, 0.072397, 0.129021, -0.213191, 0.013111, -0.055080, -0.035826, 0.057019, -0.077969, -0.272989, 0.194456, -0.130023, -0.141100, -0.063094, 0.053535, 0.217768, -0.264266, 0.209219, -0.335595, -0.290558, -0.195373, 0.115569, 0.148736, -0.271566, 0.395321, 0.016919, -0.113056, -0.258476, -0.299895, -0.048642, -0.004854, -0.141312, -0.064143, 0.045666, 0.201555, 0.008749, -0.017615, -0.047296, -0.045917, -0.123247, 0.150168, -0.221127, -0.136101, -0.059238, 0.041293, 0.293279, -0.289580, -0.062897, -0.029481, -0.058856, 0.052327, -0.131987, 0.224719, -0.545771, 0.047911, -0.015720, 0.137184, -0.238583, 0.280831, 0.044336, -0.049852, 0.123101, 0.027889, 0.119870, -0.028111, -0.014837, 0.019018, -0.095923, -0.073872, -0.241465, 0.125256, -0.010258, -0.183696, 0.067011, 0.278762, -0.014625, -0.131624, 0.011466, -0.128793, -0.036122, -0.040780, 0.188463, 0.021049, -0.114170, 0.009398, -0.028687, -0.133077, 0.075354, 0.016099, -0.153854, 0.238736, -0.212811, -0.086779, -0.196763, 0.224163, 0.314640, 0.036482, -0.196946, -0.104812, -0.253971, -0.261948, 0.002037, -0.126382, -0.358360, 0.387414, 0.132928, -0.055380, -0.058644, -0.500478, -0.042084, 0.078546, -0.098719, -0.123120, 0.067539, -0.039376, 0.113115, -0.165327, 0.001359, -0.083317, -0.045851, -0.156556, -0.159937, -0.281242, -0.073873, 0.101351, 0.128057, -0.141840, 0.080926, -0.477456, 0.005035, 0.117924, 0.177581, 0.167223, -0.496414, -0.152296, 0.292274, -0.564504, -0.237597, -0.003863, 0.104995, -0.122719, 0.040292, 0.009678, -0.080685, -0.067238, -0.119641, -0.054403, -0.008779, 0.019508, 0.222147, 0.135021, -0.225510, -0.167295, -0.082933, 0.131926, 0.212116, -0.008292, -0.063888, -0.053919, -0.045049, -0.015514, -0.054554, 0.194939, -0.176957, 0.004614, 0.059372, -0.573772, -0.121996, -0.017217, -0.188855, -0.010324, -0.220712, -0.087896, -0.181688, 0.257079, 0.281163, -0.251266, -0.042767, -0.083084, -0.158501, -0.127896, -0.159575, -0.275380, -0.192674, -0.033462, 0.135634, -0.205858, 0.208618, -0.169190, -0.055577, 0.086925, 0.043698, -0.120136, 0.037911, 0.066490, 0.019154, -0.183402, 0.063251, -0.255822, -0.179033, 0.006749, -0.040200, -0.336956, 0.229885, -0.007979, 0.222583, -0.552513, -0.171096, -0.596550, -0.049887, 0.100796, -0.053732, 0.126321, 0.087686, 0.196303, 0.058107, -0.495466, -0.228115, -0.462627, -0.075198, 0.107574, 0.098095, 0.051521, 0.054253, -0.085803, -0.054535, -0.165489, -0.004299, -0.080083, -0.121143, -0.020427, 0.061320, -0.211960, -0.145606, 0.156683, 0.143154, -0.392273, 0.092505, -0.107140, -0.198153, -0.016893, -0.059509, 0.036828, 0.060385, 0.069688, 0.107336, -0.350398, -0.074266, -0.512670, 0.134154, -0.154770, -0.023956, -0.100977, -0.115817, 0.105887, 0.003104, -0.441684, 0.017568, -1.314877, -0.000375, -0.113475, -0.062140, -0.195757, -0.025797, 0.044517, 0.196141, -0.078326, -0.006671, -0.174065, -0.122164, 0.051581, -0.030479, -0.104459, 0.143148, 0.072644, -0.002965, -0.165748, -0.216595, -0.248841, -0.261642, -0.325857, -0.205957, -0.392106, 0.168085, 0.061721, 0.260629, -0.268023, -0.005820, -0.350386, -0.062849, -0.295105, 0.100283, -0.319388, 0.243264, 0.109824, 0.052185, -0.444407, -0.150552, -0.242965, 0.098146, 0.064373, -0.046029, -0.010182, 0.205768, 0.077986, -0.015828, -0.165924, -0.230244, -0.378976, -0.140983, 0.032457, 0.035665, -0.032900, 0.224403, 0.001374, 0.051669, -0.644306, -0.126350, -0.965245, -0.092848, -0.088869, -0.133321, -0.154159, 0.156732, 0.022936, 0.174846, -0.545935, 0.003996, -1.206669, -0.238902, 0.114280, -0.134407, -0.076587, -0.313889, 0.396050, -0.248907, 0.228548, -0.371170, -0.236953, -0.251358, -0.279176, -0.040479, 0.334177, -0.069282, -0.149444, 0.218537, -0.186495, -0.035871, -0.441473, 0.152755, 0.252743, -0.010591, 0.011300, 0.087164, 0.021372, 0.140345, -0.090213, -0.157621, -0.283461, -0.226934, -0.027383, 0.015699, 0.060774, -0.091414, 0.042668, 0.255127, -0.156310, -0.061148, -0.690296, -0.244451, -0.249565, 0.184652, -0.065585, 0.234888, -0.108877, -0.086819, -0.264215, 0.057967, -0.378521, 0.183613, 0.227560, -0.040620, -0.246416, 0.084311, -0.177812, 0.024430, 0.114858, -0.115604, -0.003696, -0.288201, 0.141975, -0.000882, -0.005957, -0.065208, -0.142258, -0.075567, -0.142926, -0.276819, -0.364085, -0.172270, 0.104740, 0.163542, -0.157107, -0.013263, -0.300825, -0.079930, -0.010159, -0.064591, -0.516079, -0.254809, -0.063520, -0.263557, 0.032104, -0.239345, 0.327264, 0.215991, 0.184895, -0.440744, -0.163485, 0.022350, 0.041127, -0.074985, 0.065144, -0.206436, -0.214392, 0.037752, 0.003726, 0.101608, -0.416106, 0.166475, -0.043502, 0.102439, -0.004612, -0.036446, -0.097062, 0.016485, -0.064870, 0.000139, -0.070281, 0.055438, -0.048230, 0.037404, 0.078388, -0.188773, -0.086722, 0.204434, -0.007814, -0.018535, -0.076052, 0.035500, -0.140090, 0.118949, 0.175376, -0.043349, -0.311833, -0.135405, -0.085476, 0.045575, -0.017844, 0.048915, 0.173197, 0.003192, -0.122465, -0.080108, 0.038989, 0.203906, 0.056245, -0.198516, -0.066624, -0.004487, -0.047731, 0.027970, 0.033203, -0.224925, -0.192296, 0.119582, 0.011951, 0.101579, -0.268943, -0.196569, 0.081657, 0.006277, 0.101712, -0.044010, 0.079192, 0.102449, -0.004234, -0.194543, -0.279362, -0.265465, -0.245986, -0.006551, 0.101650, -0.269798, -0.118908, 0.554938, -0.186739, -0.634221, -0.876435, -0.066349, -0.102337, -0.117627, 0.108641, 0.066103, -0.123429, -0.039819, -0.075710, 0.105212, -0.232003, 0.010816, -0.141496, 0.045322, 0.110723, -0.060195, -0.015823, -0.029991, 0.052359, 0.108414, 0.079935, -0.035196, -0.314283
.float 0.068313, 0.017074, -0.157437, -0.119788, 0.225013, -0.048187, 0.116752, -0.025458, -0.003090, -0.137394, 0.014658, -0.070701, -0.236673, -0.050388, -0.089374, 0.140685, 0.041492, 0.136837, -0.091940, 0.255192, -0.124661, -0.132084, -0.033835, 0.077425, -0.186370, 0.181701, -0.276590, -0.005304, -0.061210, -0.119667, -0.062886, 0.051350, 0.199007, -0.005500, -0.009450, -0.108420, 0.048589, -0.181747, -0.024360, 0.108378, -0.025255, 0.058532, -0.070786, -0.067938, -0.055433, 0.008849, -0.098435, -0.064583, -0.338674, -0.215749, -0.304401, -0.034603, 0.132677, -0.212092, 0.334918, 0.021126, 0.318989, -1.184309, 0.045683, 0.025014, 0.078044, -0.019817, -0.296225, 0.131252, 0.127430, 0.016699, -0.163162, -0.012360, 0.050354, -0.012947, -0.117988, 0.053209, -0.151580, -0.001092, -0.016950, -0.023091, -0.048674, 0.084786, 0.039672, 0.021435, 0.055911, 0.072185, -0.349558, 0.045263, 0.243160, -0.100838, -0.033519, 0.008095, 0.173724, 0.067279, -0.011051, 0.060461, -0.379377, -0.093287, 0.058783, -0.213699, -0.046505, 0.340725, -0.079787, 0.128413, -0.101852, -0.012692, 0.216745, 0.002590, -0.555058, 0.131448, -0.181974, -0.060435, 0.003949, -0.208571, -0.022060, 0.063879, -0.122546, 0.144927, -0.128635, -0.109802, 0.067039, 0.112777, -0.214489, -0.023268, -0.098070, 0.072022, 0.128334, -0.012797, -0.028154, 0.027027, -0.129209, -0.101471, -0.165778, -0.059891, -0.264747, -0.129672, 0.515665, -0.038796, 0.274137, -0.148074, 0.203686, -1.147736, -0.067227, -0.041353, 0.072270, 0.119190, -0.276292, -0.145783, -0.052488, 0.034907, 0.089264, -0.041026, 0.034110, -0.119712, -0.133097, 0.039780, -0.109753, 0.078361, -0.018376, 0.100521, 0.081159, 0.028963, -0.092227, 0.235388, 0.003579, -0.037685, -0.006756, -0.015797, -0.005989, -0.082434, -0.081452, -0.107889, 0.066939, -0.180924, 0.046374, 0.107346, -0.368819, 0.012347, -0.176927, -0.049882, -0.042820, 0.110521, -0.054346, -0.093032, -0.071999, 0.157497, 0.336841, -0.103703, -0.597769, 0.145378, -0.154148, -0.135422, -0.006439, -0.098360, 0.104748, 0.173375, -0.443731, -0.174173, -0.132339, 0.004063, 0.014786, 0.259189, -0.127805, 0.085700, -0.017199, 0.075488, 0.083689, 0.042117, -0.176005, 0.010381, -0.061682, -0.288294, -0.132289, 0.185666, -0.136981, -0.054553, 0.523765, -0.121840, 0.375884, -0.609045, -0.268762, -0.475905, 0.053480, -0.027978, 0.078784, 0.064472, -0.097368, -0.107698, -0.164709, 0.031813, -0.048627, 0.017004, -0.087520, 0.020702, -0.152171, 0.021264, -0.140866, 0.025436, -0.046886, 0.066631, 0.141384, 0.185705, 0.051167, 0.211510, -0.151232, -0.095207, 0.031938, 0.058256, 0.223347, -0.164374, 0.028974, -0.133135, -0.024152, 0.079931, -0.050596, 0.176698, -0.195584, -0.044621, -0.194094, -0.063580, -0.089161, 0.076113, -0.211576, -0.239148, 0.006857, 0.234516, 0.072219, 0.007441, -0.487121, 0.075023, -0.148306, -0.132466, -0.019983, 0.052572, 0.042722, 0.220785, -0.371976, -0.172091, -0.008616, -0.017915, 0.080126, 0.139697, -0.077682, -0.082251, 0.109919, 0.213729, -0.043348, -0.116305, -0.190067, 0.113791, -0.032775, -0.055284, -0.006987, 0.413757, -0.435975, -0.188117, 0.492844, 0.000655, 0.068552, -0.281712, 0.067724, -0.385302, 0.050704, -0.004998, 0.095551, 0.096623, -0.362392, -0.066591, -0.119287, 0.124558, 0.059380, -0.025446, -0.098390, -0.061684, -0.265559, -0.015217, -0.239462, 0.108116, -0.017857, 0.009992, 0.038738, 0.125760, -0.035507, 0.166266, -0.054441, -0.046227, 0.007368, 0.156575, 0.089146, -0.351789, -0.045173, 0.076627, 0.055268, -0.056236, 0.034224, 0.126916, -0.363801, 0.034644, -0.259347, 0.051050, -0.063997, 0.007644, 0.022514, -0.270666, 0.098018, 0.174478, -0.017135, -0.092740, -0.511038, 0.180755, -0.147951, -0.092535, 0.031708, 0.033831, 0.116210, -0.072358, -0.360943, 0.009512, -0.062965, 0.045485, -0.003125, 0.127693, 0.050703, -0.105402, 0.031621, 0.050944, -0.088856, -0.100841, -0.168293, 0.048409, -0.043560, 0.005842, -0.253229, 0.316714, -0.385062, -0.053000, 0.366334, 0.209629, -0.001838, -0.192643, -0.026080, -0.204329, -0.036147, -0.007108, -0.090924, -0.006495, -0.195871, 0.064616, -0.026625, -0.015970, 0.161720, 0.033199, -0.031774, 0.032255, -0.219737, -0.094820, -0.050625, 0.187368, 0.068737, 0.101177, 0.007182, 0.132386, -0.089398, 0.025045, -0.071816, -0.246504, 0.066331, 0.300647, -0.033235, -0.216118, 0.078060, -0.046218, 0.087113, -0.054140, 0.000765, 0.052489, -0.275160, 0.046180, -0.287702, 0.083624, 0.010434, -0.078423, -0.020961, -0.124875, 0.054878, 0.096891, -0.077077, 0.021390, -0.131063, 0.012798, 0.006387, -0.017701, -0.050095, 0.003526, -0.032334, -0.234138, -0.180259, 0.020806, 0.039677, -0.024560, 0.220978, 0.133672, -0.047896, -0.059262, 0.068833, 0.142563, -0.100403, -0.057747, -0.127919, -0.012609, -0.004852, 0.111873, -0.263647, 0.143133, -0.045622, -0.079685, 0.249704, 0.213993, -0.101843, -0.053388, -0.047078, -0.194759, -0.107783, -0.224134, 0.054877, 0.043422, -0.135504, 0.033105, 0.063602, -0.119736, -0.127338, 0.210756, -0.137813, -0.021439, -0.174211, -0.200801, 0.084919, 0.214977, 0.012516, 0.128476, 0.047347, 0.061924, -0.072147, 0.146141, -0.261011, -0.086317, 0.309280, 0.124344, 0.089164, -0.368491, 0.077638, -0.116598, 0.136432, -0.195449, 0.053544, -0.074834, -0.214847, 0.117414, -0.182187, 0.165819, -0.015535, -0.041278, 0.010880, -0.209677, -0.032783, -0.074497, -0.009005, 0.140696, -0.054187, -0.000807, 0.034689, 0.032359, -0.008967, -0.234665, 0.131788, -0.192244, -0.158623, 0.164695, 0.004181, -0.094573, 0.108170, 0.075158, -0.027323, -0.131514, 0.073644, 0.090077, -0.042145, -0.079620, -0.188028, 0.012733, 0.052304, 0.039224, -0.080815, 0.229602, -0.271123, -0.107579, 0.115253, 0.150292, 0.041342, -0.100301, 0.057771, -0.073722, 0.015339, -0.115282, 0.075863, 0.134349, -0.298524, -0.043405, 0.188982, 0.022063, -0.084554, -0.003933, -0.079455, 0.022331, -0.152189, -0.160357, 0.054544, 0.110376, -0.093294, 0.061730, 0.043299, -0.112629, 0.239472, 0.023419, -0.390991, -0.453802, 0.108432, 0.141958, -0.116575, 0.195506, 0.175099, -0.095146, 0.221656, 0.135464, 0.011004, -0.133295, -0.052286, -0.046325, -0.056625, 0.163410, -0.026690, -0.091836, -0.022640, -0.134195, -0.058572, -0.090508, -0.009143, 0.043187, -0.112306, 0.016010, 0.160064, 0.019773, -0.070236, 0.116477, -0.234068, -0.153682, 0.021997, 0.138906, 0.049978, -0.023132, -0.035127, -0.028847, -0.019844, -0.079169, -0.024116, 0.121571, -0.044195, 0.024685, -0.017871, -0.020132, -0.018448, 0.051520, -0.172325, -0.029514, -0.013762, -0.171089, 0.129868, 0.134756, 0.036890, -0.027960, -0.036231, -0.041458, -0.209119, -0.145319, -0.268064, 0.386137, 0.086401, 0.017440, 0.036954, -0.453165, 0.013251, -0.476230, -0.156025, -0.056285, -0.073616, -0.290221, 0.132325, 0.264965, 0.005302, -0.051760, -0.019159, -0.088668, -0.057659, 0.229718, -0.265755, -0.437324, 0.162331, 0.095901, -0.127216, -0.360348, -0.076761, -0.251815, 0.103181, -0.157548, -0.129056, -0.109034, -0.183559, 0.026234, 0.039751, -0.146655, -0.156348, 0.393695, -0.065264, 0.047435, 0.024275, -0.150463, -0.129303, -0.060667, -0.012198, 0.057546, 0.018961, 0.070791, -0.094559, 0.061775, -0.100842, -0.273748, 0.167997, 0.235336, 0.041353, 0.057234, -0.088068, -0.168295, 0.049724, -0.107562, 0.051632, 0.009453, -0.015610, -0.232112, 0.006665, -0.032242, 0.079675, 0.122870, 0.058929, 0.025150, 0.018889, -0.045110, 0.161213, -0.043705, -0.117393, -0.085304, -0.084560, -0.241008, -0.187204, -0.289622, -0.038773, -0.160962, 0.176362, -0.078507, 0.081118, -0.164316, 0.065801, -0.184255, -0.024024, -0.097483, -0.298202, -0.422809, 0.100027, 0.285310, -0.097653, -0.177212, 0.029458, -0.077085, -0.389270, -0.328095, -0.044712, -0.315122, -0.104642, 0.125490, 0.191782, -0.614622, 0.198932, -0.431159, 0.127909, -0.036141, -0.017782, 0.056938, -0.007553, 0.020254, 0.099179, -0.549451, -0.108311, -0.192589, 0.225738, -0.040164, -0.171561, -0.071903, 0.005547, -0.012385, -0.010125, -0.073629, -0.076389, 0.031677, -0.110095, -0.195902, -0.206874, -0.084516, 0.136677, 0.018275, -0.013205, -0.095528, 0.083790, -0.145036, 0.130370, -0.233773, -0.137428, 0.051643, -0.005769, 0.022456, -0.040321, -0.415217, 0.071660, -0.005135, -0.082249, -0.033208, 0.204876, -0.087378, -0.379635, 0.300524, -0.425526, 0.161171, -0.481884, -0.543852, -0.311182, -0.017158, 0.146003, 0.034395, -0.059067, -0.241778, 0.131706, -0.217274, 0.216612, -0.486120, 0.085180, 0.029888, -0.020533, -0.077390, 0.087584, -0.097572, -0.118689, 0.050432, -0.073620, -0.056475, -0.371573, 0.004694, -0.009105, 0.060963, -0.048430, 0.020014, 0.237690, -0.064680, -0.011319, -0.096048, -0.387057, 0.018072, 0.006827, 0.073931, -0.159116, -0.254331, 0.039196, -0.102170, 0.006123, -0.235335, 0.152168, 0.344520, -0.158975, -0.048899, -0.011497, -0.143522, -0.043845, 0.212516, -0.279649, -0.095512, -0.132166, -0.239056, -0.027125, 0.113994, 0.048238, -0.099626, 0.049319, 0.097499, 0.099949, -0.300511, -0.053224, -0.260406, -0.278302, 0.177032, 0.259116, -0.234015, -0.129603, 0.175497, -0.031348, -0.392350, -0.415810, -0.639749, 0.283776, 0.061286, -0.339578, -0.244196, 0.267499, -0.037568, -0.309629, -0.473613, 0.177690, -0.041314, -0.080826, -0.004431, 0.152852, -0.000281, -0.358352, -0.077168, 0.085686, 0.042151, 0.083042, 0.131630, -0.017631, 0.015447, -0.044445, -0.089184, -0.127493, -0.020647, 0.005347, 0.021066, 0.139232, 0.039185, -0.195433, 0.082455, -0.191563, -0.092162, -0.028808, -0.069539, 0.012801, 0.008104, 0.167758, -0.041953, -0.318708, 0.175549, 0.053044, 0.063926, -0.357230, 0.075873, -0.138337, -0.028517, -0.131193, 0.243778, -0.097854, 0.091846, 0.188990, -0.295292, -0.099134, 0.286570, -0.301834, -0.157383, -0.092415, -0.200984, 0.045328, 0.067873, 0.124569, -0.083084, 0.052107, -0.046535, -0.001088, -0.032496, -0.166905, -0.023027, -0.019790, 0.111385, 0.113217, -0.254298, 0.008844, 0.107336, -0.188944, -0.279849, -0.304171, -0.298810, 0.164079, 0.047263, 0.227041, -0.125857, 0.281669, -0.452286, -0.009490, -0.811337, -0.101817, -0.285654, 0.076507, -0.077714, 0.024163, 0.224865, -0.012909, 0.191885, -0.031061, -0.085152, 0.108558, -0.152220, -0.077304, 0.086324, -0.125439, -0.062934, 0.020348, -0.020748, 0.002629, 0.154616, -0.032608, -0.132910, -0.069796, 0.062013, -0.022989, 0.109098, 0.200113, -0.033184, 0.008847, -0.083903, 0.140042, -0.059867, -0.056464, 0.087145, 0.027020, 0.136721, -0.100888, -0.024822, -0.121125, -0.242504, -0.103254, 0.328374, -0.054790, 0.086076, 0.044575, -0.294794, -0.315072, 0.335112, -0.159902, -0.192712, 0.074271, -0.107720, 0.008970, -0.063566, -0.307562, 0.071821, -0.162155, -0.074534, 0.104001, 0.302436, -0.072640, 0.297057, -0.188968, -0.021954, -0.151354, 0.026711, 0.156903, 0.274823, -0.053031, -0.241337, -0.190494, -0.416254, -0.086943, -0.118277, 0.277328, -0.172815, 0.375685, -0.073239, 0.151723, 0.103868, 0.058658, 0.086363, 0.069398, -0.014956, -0.317031, 0.098457, -0.166384, 0.110687, -0.011032, -0.117724, 0.007510, -0.262595, 0.168290, 0.130881, -0.065304, -0.214081, -0.068916, 0.031401, -0.072747, 0.108235, 0.048322, 0.104791, 0.061303, -0.025075, 0.016518, 0.180233, 0.210763, -0.019193, -0.139737, 0.017320, 0.053144, 0.251098, 0.029182, 0.118342, -0.295601, 0.152936, -0.008307, 0.075335, -0.089637, -0.117439, -0.139196, 0.242530, -0.125655, 0.123945, 0.111862, -0.150198, -0.600337, 0.287983, -0.186525, -0.347066, -0.098927, 0.031589, -0.078104, 0.091869, -0.275467, 0.023815, -0.041878, -0.026668, -0.040647, 0.159851, -0.128250, 0.252319, -0.166132, 0.080232, 0.144598, -0.015776, -0.168226, 0.201707, -0.059440, -0.387546, -0.240429, -0.351816, -0.082904, -0.213672, 0.311978, -0.081682, 0.349158, -0.056489, -0.130168, 0.075373, 0.104007, -0.074135, 0.096098, 0.034875, -0.228098, 0.235585, -0.147051, 0.024847, 0.040046, -0.205930, -0.033851, -0.091929, 0.134167, -0.021815, -0.225717, -0.141311, -0.068148, 0.052755, -0.032602, 0.114431, -0.105522, 0.181052, -0.043518, -0.162001, 0.184277, 0.224640, 0.002704, 0.073475, -0.002170, -0.124099, 0.008184, 0.022538, -0.067628, 0.146392, -0.306853, 0.116353, -0.125588, 0.284291, -0.057196, -0.113129, -0.424427, -0.408496, 0.064441, 0.247266, -0.111047, -0.144295, -0.840172, 0.402698, -0.010931, -0.067966, -0.117807, 0.129661, -0.009318, 0.232660
.float -0.261655, -0.287823, -0.179203, -0.137620, 0.060992, 0.110346, -0.173912, 0.108001, -0.026336, 0.262783, 0.256363, -0.029401, -0.180458, 0.300087, -0.126059, -0.503077, -0.013504, -0.008611, 0.041739, -0.277523, 0.177014, 0.049910, 0.146153, -0.185604, -0.003617, -0.044538, 0.130908, 0.065785, 0.201094, 0.038997, -0.336434, 0.015284, -0.201877, 0.022244, -0.050375, 0.208669, 0.077279, -0.049939, 0.024287, -0.076867, -0.289391, -0.021683, -0.105411, 0.112589, -0.087685, 0.248298, -0.045496, 0.193606, -0.113685, -0.154379, 0.198062, 0.272734, 0.017973, 0.022787, -0.103625, -0.157364, 0.079918, 0.051337, -0.093913, 0.069936, -0.251578, 0.059867, -0.227365, 0.211591, 0.016221, 0.045345, -0.097276, -0.618271, -0.013080, 0.201642, -0.282945, 0.008710, -0.733107, 0.342208, 0.104284, 0.174770, 0.033006, 0.048003, 0.025857, 0.214553, -0.403815, -0.233554, -0.063365, -0.174760, 0.054839, 0.076034, -0.111215, 0.017197, 0.072363, 0.246110, 0.143611, -0.212263, -0.439067, 0.102114, -0.132829, -0.187949, -0.040324, 0.035161, 0.093955, -0.192454, 0.417514, -0.000880, 0.167854, -0.142484, -0.012606, -0.244511, 0.094513, -0.084726, -0.060454, -0.028152, -0.240304, 0.126272, -0.277594, 0.050300, 0.118170, 0.092662, 0.010980, 0.227492, -0.095112, -0.238213, -0.168714, 0.147408, -0.027892, 0.058188, -0.124027, 0.032667, -0.030114, 0.202776, -0.159469, -0.062971, 0.367574, 0.044088, 0.046440, -0.167243, -0.145082, -0.133065, 0.056241, -0.057343, -0.201395, 0.136467, -0.487725, 0.185078, -0.065416, 0.186997, 0.033467, 0.037249, -0.010957, -0.608644, -0.000941, 0.103544, -0.674044, 0.144105, -0.487750, 0.075863, 0.166389, 0.366795, -0.058372, 0.009425, 0.003296, 0.132436, -0.234008, -0.072324, -0.078510, -0.114442, 0.134944, 0.119278, 0.031083, 0.204682, 0.184930, 0.126903, 0.040826, -0.366950, -0.423179, 0.065354, -0.127827, 0.070361, -0.324292, 0.024283, -0.118401, -0.186799, 0.377992, -0.095244, 0.108763, -0.064122, -0.048519, -0.109546, 0.289158, -0.051048, 0.032094, -0.070126, -0.274497, -0.034969, -0.155291, 0.200634, -0.009968, -0.105340, -0.151540, 0.053110, -0.461061, -0.254264, 0.029177, 0.353183, 0.124473, 0.123643, -0.178377, -0.015086, -0.087557, 0.201920, -0.203230, -0.106622, 0.469861, 0.096218, -0.228920, -0.159610, -0.028947, -0.147574, -0.067174, 0.029134, -0.023294, -0.050685, -0.328796, 0.021277, -0.124880, 0.281813, -0.111404, 0.159001, 0.068116, -0.676539, 0.067380, -0.043072, -0.464439, 0.059794, -0.108267, 0.000642, 0.136367, 0.427828, 0.067313, -0.288827, 0.013439, -0.197017, -0.277479, -0.085346, -0.012652, -0.057748, 0.269010, 0.059344, 0.116581, 0.023859, 0.073516, 0.036873, -0.068294, -0.273030, -0.319431, 0.145027, -0.046039, 0.041556, -0.185453, 0.009311, -0.086298, -0.044769, 0.334715, 0.260887, -0.011111, 0.012201, -0.057644, -0.247789, 0.093147, -0.018670, -0.063252, 0.063198, -0.168276, -0.075809, -0.152091, 0.061987, 0.095678, -0.005529, -0.340968, 0.327317, -0.471761, -0.448401, 0.138398, 0.484554, 0.168557, 0.241151, -0.086534, -0.236709, -0.041539, 0.344990, -0.278540, -0.027990, 0.383171, -0.303572, -0.048690, -0.538515, 0.252858, -0.231726, -0.117529, -0.029986, 0.083897, -0.136464, -0.216825, 0.097639, -0.142563, 0.211178, 0.091895, 0.002937, 0.040942, -0.571272, 0.049863, -0.096936, -0.283775, 0.138453, -0.073819, -0.025205, 0.202730, 0.089554, 0.007346, -0.069377, 0.089255, -0.254866, 0.070738, 0.058230, -0.133247, -0.078616, 0.216810, 0.074828, 0.164654, -0.067663, 0.173317, -0.112371, -0.044764, -0.138070, -0.420442, 0.196107, 0.032409, 0.000094, -0.236336, 0.045428, -0.149143, -0.024745, 0.189815, 0.177209, -0.113697, 0.032877, 0.041795, -0.092615, 0.098369, 0.049729, -0.010268, 0.093523, -0.263505, -0.120423, -0.127165, -0.025824, 0.023142, 0.141651, -0.275482, 0.165148, -0.313115, -0.347640, 0.202526, 0.580000, 0.161724, 0.010203, -0.120017, -0.274588, -0.057261, 0.195790, -0.232932, -0.456181, 0.203024, 0.160919, -0.069380, 0.036012, 0.079996, -0.243989, 0.170868, -0.093896, 0.028812, 0.118190, -0.055032, -0.050664, -0.162328, 0.121667, -0.021857, -0.025386, 0.203133, -0.450492, -0.110689, -0.097152, -0.143808, 0.038063, -0.065835, -0.036523, 0.367729, 0.116537, 0.010967, -0.125415, -0.013069, -0.256453, 0.023437, 0.305739, -0.077114, 0.028864, -0.051478, 0.174262, 0.061699, -0.087989, 0.043248, 0.083630, -0.212943, -0.309228, -0.051071, 0.004442, 0.054629, 0.078957, -0.077652, -0.037253, 0.036090, -0.093602, 0.192559, 0.014714, -0.133154, 0.038804, -0.033914, -0.104712, 0.066187, -0.272966, -0.108965, 0.120873, 0.025664, -0.245651, 0.102401, -0.272290, 0.004869, 0.140336, -0.082933, -0.167262, -0.299152, -0.663422, 0.058235, 0.427977, -0.198079, -0.075540, 0.036758, -0.031443, -0.318386, 0.188946, -0.193394, -0.301027, 0.275859, 0.014321, -0.423604, -0.062119, 0.073123, -0.168594, 0.087107, -0.152091, -0.063862, 0.131711, 0.139871, 0.134522, -0.152735, 0.193549, -0.067213, -0.254241, 0.069944, -0.398740, -0.123368, -0.160530, -0.018182, 0.045721, -0.012438, -0.166990, 0.113420, 0.057314, -0.087138, -0.057560, -0.147103, -0.288412, 0.188242, 0.201499, 0.055579, -0.197663, -0.122802, -0.016766, 0.099905, -0.046965, -0.009524, -0.009173, -0.014728, -0.216382, -0.082005, 0.009932, 0.026314, 0.156563, 0.004950, -0.049900, -0.038675, 0.044015, 0.042814, -0.065071, -0.062561, -0.073293, 0.085596, -0.190941, 0.049551, -0.143738, 0.013940, 0.310556, 0.201904, -0.002766, 0.047636, -0.422725, -0.263987, 0.065800, -0.177444, -0.094441, 0.077063, -0.585949, 0.052692, 0.460105, -0.473245, 0.053690, -0.067295, -0.251573, -0.263973, -0.258372, -0.325354, -0.402259, -0.007550, -0.049460, -0.307576, 0.062603, 0.196762, -0.452502, 0.082806, -0.172211, -0.149160, 0.072074, -0.115357, 0.448065, -0.240195, -0.171126, 0.086237, -0.258066, 0.124555, -0.214013, -0.070612, -0.124241, -0.054408, 0.084043, -0.235273, 0.104957, 0.101867, 0.065817, 0.003378, -0.290595, 0.002505, 0.252286, 0.003534, 0.177735, -0.167924, -0.119925, -0.054274, -0.275276, 0.235858, -0.225314, -0.091045, 0.048333, -0.152635, 0.014556, -0.280154, -0.096606, 0.079734, -0.151042, 0.372222, -0.297769, -0.107670, 0.306310, -0.458831, -0.418772, 0.037281, 0.036442, -0.282023, -0.577839, -0.280264, 0.086724, -0.057500, -0.146248, -0.107806, -0.188664, 0.105228, 0.056520, 0.122528, -0.082495, -0.020032, 0.077690, 0.068531, -0.082236, -0.003155, -0.056729, -0.318876, 0.029706, -0.062596, 0.078377, -0.154065, -0.051496, 0.111533, -0.026702, -0.062686, -0.057145, 0.136451, -0.086884, 0.042658, -0.090980, -0.108823, 0.072063, -0.134399, -0.150597, -0.234939, -0.018707, -0.086915, 0.161040, 0.249899, -0.089004, 0.325144, 0.121974, -0.072886, 0.128423, 0.069273, -0.037353, -0.011611, 0.122761, -0.128653, -0.215874, -0.192950, -0.302998, -0.071551, 0.071685, 0.277319, -0.192457, 0.049611, 0.093251, -0.007050, -0.117929, 0.230314, -0.445482, -0.047497, 0.367308, -0.012755, -0.463962, -0.362441, -0.089471, -0.056398, -0.390229, -0.387568, 0.004074, 0.513182, 0.192366, -0.275242, -0.555174, 0.388097, -0.606980, -0.253765, 0.207634, 0.210524, -0.028882, -0.221989, 0.147126, 0.063196, -0.009466, -0.181929, -0.054600, -0.005576, 0.070699, 0.067428, 0.143090, 0.064169, 0.019252, -0.089594, -0.116491, 0.031235, -0.086886, -0.054074, 0.111634, 0.128737, -0.042112, -0.100696, -0.066913, -0.086885, -0.026815, -0.098525, 0.036753, -0.006229, 0.135740, 0.156994, -0.246167, 0.046547, 0.026987, 0.043327, -0.023404, -0.292716, -0.025676, -0.063220, -0.047106, 0.008491, -0.054253, 0.105808, 0.234374, 0.147613, -0.445538, -0.065787, 0.129016, -0.119895, -0.283464, -0.010335, 0.188559, -0.214380, 0.111364, -0.062402, -0.014071, -0.087618, -0.029909, 0.116120, 0.102575, -0.127513, 0.078045, 0.239067, 0.248592, 0.397664, -0.173101, -0.012982, 0.044126, -0.452889, -0.601139, 0.041603, -0.323096, 0.177987, -0.199787, -0.019847, -0.231778, 0.078284, -0.317619, 0.137414, 0.374259, 0.017170, 0.170907, -0.298319, 0.077039, -0.019066, 0.095038, -0.080901, 0.162312, -0.031505, -0.062700, 0.017444, -0.074660, 0.137924, 0.039688, 0.005937, -0.203378, -0.016565, -0.005141, 0.042647, -0.020123, 0.009597, -0.235666, -0.148371, -0.181774, 0.132649, 0.165033, 0.214246, -0.016079, 0.051978, -0.020477, -0.060660, -0.236951, -0.039740, -0.220188, -0.122121, 0.094020, -0.161557, 0.298241, 0.099770, -0.084066, -0.049375, -0.080021, 0.081733, 0.121875, 0.013103, -0.452959, -0.510528, 0.378671, 0.121665, -0.347458, 0.045295, -0.074857, -0.040813, -0.023141, -0.225092, 0.062944, 0.213661, 0.024959, -0.090045, 0.052366, 0.060385, 0.042614, -0.111327, -0.217555, 0.172060, -0.163690, -0.079811, 0.465164, -0.173588, -0.536472, -0.131509, -0.243119, -0.150408, 0.022345, 0.198078, -0.254179, 0.211679, -0.338941, 0.092336, 0.382275, -0.015389, 0.005299, -0.114725, -0.030641, -0.292823, 0.042551, -0.096360, 0.271100, -0.098465, -0.131099, 0.037356, -0.185806, 0.128248, 0.181340, 0.009144, -0.217198, -0.040069, -0.121873, -0.045758, 0.053701, 0.045363, 0.008440, -0.147711, -0.127987, 0.052190, 0.069914, 0.081057, 0.075244, -0.030345, -0.076626, 0.034574, -0.008084, -0.197100, 0.004541, -0.095323, -0.024766, 0.063159, 0.208216, 0.009755, 0.065334, -0.191556, 0.152196, 0.104051, 0.224836, -0.273441, -0.213607, -0.820002, 0.373209, -0.159011, -0.154710, 0.060715, 0.260369, -0.144863, 0.069166, -0.125931, -0.031692, 0.121977, -0.107789, -0.103019, 0.092272, -0.106602, 0.217238, -0.413988, -0.371432, 0.360485, -0.050858, -0.087055, 0.316649, -0.019866, -0.367509, 0.038946, -0.210705, -0.009550, -0.123167, 0.107998, -0.184358, 0.158957, -0.064966, -0.051145, 0.139297, -0.091590, -0.074551, -0.100852, 0.047619, -0.238064, 0.057289, -0.154399, 0.178201, 0.127642, -0.095186, 0.070382, -0.234967, 0.065505, 0.310944, -0.122731, -0.161655, -0.044614, 0.032917, -0.195406, 0.131198, -0.023320, 0.013659, -0.226626, -0.095149, 0.238802, 0.037964, -0.044022, 0.022905, -0.117989, -0.019661, -0.099653, -0.001449, -0.065914, -0.007551, 0.006669, 0.085330, 0.001343, 0.037868, 0.022707, -0.024110, -0.169971, -0.321517, -0.090615, 0.070521, -0.497409, -0.054840, -0.555327, 0.217658, 0.213840, -0.023026, -0.138661, 0.064651, -0.187016, 0.324914, -0.084695, -0.064939, -0.170660, -0.133899, 0.255058, 0.154177, -0.026222, 0.089521, -0.278573, -0.399442, 0.347549, -0.017051, -0.063803, 0.267593, -0.093825, -0.187892, 0.042333, -0.397782, 0.072354, 0.015335, 0.106093, -0.107550, 0.181650, -0.101037, -0.040743, 0.137423, 0.012165, 0.091028, -0.022066, -0.081792, -0.354272, -0.040936, -0.304436, 0.173060, 0.051680, 0.097698, 0.177263, -0.051050, -0.103943, 0.219130, -0.320304, -0.119953, 0.084652, 0.150406, -0.221894, 0.064200, -0.030982, 0.182149, -0.112149, 0.085477, 0.211254, 0.109704, -0.087778, 0.171906, -0.345908, -0.039427, 0.060704, 0.108789, -0.262861, 0.058219, -0.191041, -0.007891, -0.145689, 0.194354, -0.018183, 0.126665, -0.018532, -0.442534, 0.142914, 0.008732, -0.877544, 0.140418, -0.367455, 0.124052, 0.025563, 0.221199, -0.181863, 0.158994, 0.002846, 0.123583, -0.136376, -0.251759, -0.150757, -0.235850, 0.217173, 0.201344, -0.038102, -0.060832, -0.020089, -0.319696, 0.152385, 0.103510, -0.148561, 0.193654, -0.002423, -0.217194, -0.020347, -0.154093, -0.018320, 0.141267, 0.157214, -0.243965, 0.244668, -0.227719, -0.003088, 0.011795, 0.042887, 0.044611, -0.010300, -0.016087, -0.158388, 0.042389, -0.365894, 0.131924, 0.078078, -0.024961, 0.094954, 0.307626, -0.100488, -0.069615, -0.296661, -0.038772, 0.260454, 0.230734, -0.166391, 0.046625, -0.095056, 0.314417, 0.039499, -0.127435, 0.464126, -0.156357, -0.059793, -0.136883, -0.164322, -0.133620, 0.051448, 0.159777, -0.409574, 0.145204, 0.053978, -0.007856, -0.246736, 0.094964, 0.046464, 0.204878, -0.125426, -0.645527, -0.000199, -0.097657, -0.863091, 0.410718, -0.038320, 0.130484, 0.195970, 0.134286, -0.045933, -0.103819, -0.035811, 0.214217, -0.065675, -0.171019, -0.016577, -0.168784, 0.179032, 0.265802, -0.087742, 0.272057, -0.081313, -0.216203, 0.064914, -0.178689, -0.368850, 0.149224, 0.116502, -0.160479, -0.040106, -0.110234, -0.010850, 0.160829, 0.090020, -0.285246, 0.111500, 0.079421, 0.100177, 0.018764, 0.052349, 0.050817, -0.119287, 0.042231, -0.028460, 0.089665
.float -0.198511, 0.272040, -0.127251, 0.018152, -0.225768, 0.239665, -0.499654, -0.191437, -0.039439, 0.322101, 0.380002, 0.197942, -0.124442, -0.295891, -0.089809, 0.202086, 0.109285, -0.152733, 0.312032, -0.517362, -0.205553, -0.015101, 0.040326, 0.082199, 0.022253, 0.117909, -0.479007, 0.050034, 0.033144, -0.009148, -0.150892, 0.148033, 0.111770, 0.132178, 0.008317, -0.554668, 0.020184, -0.146740, -1.034374, 0.355190, 0.215477, 0.037076, 0.043011, 0.172184, 0.011721, -0.276377, -0.002819, 0.240439, -0.215704, -0.317036, -0.055765, -0.001296, 0.205765, 0.121333, 0.131126, 0.108232, 0.211805, -0.152509, 0.121087, -0.258949, -0.259769, 0.086267, -0.017001, 0.061998, -0.140905, -0.064285, -0.041931, 0.164111, 0.325094, -0.123192, -0.228249, 0.069516, 0.019060, -0.033199, 0.106006, 0.184810, -0.035958, -0.044497, -0.151102, 0.101755, -0.079217, 0.111829, -0.057037, -0.033978, -0.244819, 0.239802, -0.476733, -0.566193, 0.061508, 0.515424, 0.236231, 0.302291, -0.266173, -0.332663, 0.056384, 0.100143, 0.126126, -0.114084, 0.282361, -0.837418, -0.277741, -0.065601, 0.024657, 0.179949, 0.048175, 0.097371, -0.061336, 0.122702, -0.224950, -0.204914, -0.256186, 0.192226, -0.031586, -0.070966, -0.001826, -0.442708, -0.052281, -0.231171, -0.555504, 0.548514, 0.244381, 0.035927, -0.018703, 0.004718, -0.234527, -0.254637, -0.017402, 0.101725, -0.142387, -0.325297, 0.012132, -0.370530, 0.261256, 0.387228, 0.052743, -0.156344, 0.098706, -0.127681, 0.020451, -0.153906, -0.304666, 0.149156, 0.045821, 0.039683, -0.003094, 0.022412, -0.004851, -0.054428, 0.202962, -0.080987, -0.211052, 0.042065, -0.038549, -0.076215, 0.199489, 0.095450, 0.016768, 0.114013, -0.089526, -0.107157, -0.157200, -0.033182, 0.059810, 0.165283, -0.230609, 0.122865, -0.355081, -0.476120, 0.234271, 0.439956, 0.121306, 0.296280, -0.246150, -0.489806, 0.045776, 0.214053, 0.131356, -0.073318, 0.315712, -0.421206, -0.297298, -0.055413, 0.046092, -0.192005, 0.000508, 0.095341, -0.081552, 0.026704, -0.122069, -0.133760, -0.081462, 0.002825, -0.136882, 0.144585, 0.101607, -0.383872, -0.404974, -0.685198, -0.363304, 0.417515, 0.332096, -0.088264, 0.206659, 0.158949, -0.143535, 0.045750, 0.050594, 0.304202, -0.092296, -0.294098, -0.165679, -0.054395, 0.262141, -0.070708, 0.226235, -0.105157, -0.125965, -0.136797, -0.109624, -0.240536, -0.214814, 0.001838, 0.060341, 0.092832, 0.128665, 0.063740, 0.103613, 0.049431, 0.104659, -0.269900, -0.094191, 0.110459, -0.051544, -0.104070, 0.100211, -0.022210, 0.047171, 0.068287, -0.108671, -0.063450, -0.152874, -0.165598, 0.063630, 0.027499, -0.153145, 0.081719, -0.117021, -0.878719, -0.057932, 0.496532, -0.215707, 0.201395, -0.084612, -0.239077, 0.058149, 0.047082, -0.116718, 0.140967, 0.204332, -0.441076, -0.121976, -0.005730, 0.147454, -0.252109, 0.055107, 0.033669, -0.018483, -0.149412, -0.115682, 0.019967, -0.083930, -0.135067, 0.095649, 0.062397, 0.317680, -0.355552, -0.297164, -0.611969, -0.228577, 0.070139, 0.324891, -0.200852, 0.222499, 0.286086, -0.021124, 0.000032, -0.024812, -0.031833, 0.129972, 0.113766, -0.491747, -0.168791, 0.014858, 0.093250, 0.187759, -0.001818, -0.017365, -0.002182, -0.169063, -0.407537, 0.003294, -0.050700, 0.071774, 0.119610, 0.202785, -0.165965, 0.114593, 0.141388, -0.023985, -0.054952, -0.340767, -0.006050, 0.008459, -0.046631, 0.178989, -0.224858, -0.163114, 0.050151, -0.127334, 0.148305, -0.274296, -0.157264, 0.083414, 0.071270, -0.073916, 0.253009, 0.206519, -0.641683, -0.100486, 0.440772, -0.423803, -0.046869, -0.007625, -0.241425, 0.106945, -0.270778, -0.275357, -0.195185, -0.173552, 0.072396, -0.234243, -0.019358, 0.183786, -0.279371, -0.054927, -0.168050, -0.040557, 0.209957, -0.067352, 0.055811, -0.355402, -0.049717, 0.069755, -0.096287, 0.154111, -0.326193, -0.204673, -0.355959, -0.121608, 0.015748, 0.065807, -0.189969, 0.082561, 0.229079, 0.102238, -0.322637, -0.116388, -0.239505, 0.206761, 0.382428, -0.619126, 0.190898, -0.077307, -0.380066, 0.131182, -0.052369, 0.043223, 0.142719, -0.372585, -0.135965, -0.191119, -0.154886, 0.253582, -0.098797, 0.196963, -0.485910, 0.223341, 0.033444, -0.013158, -0.767277, -0.106308, -0.051112, -0.519213, -0.257733, -0.331600, -0.225503, -0.142595, -0.218035, -0.144627, 0.161979, 0.090438, 0.055373, 0.296080, 0.058156, 0.026109, 0.188322, -0.014961, -0.231767, 0.151271, -0.169639, -0.312342, -0.019615, 0.135252, 0.027448, -0.230962, -0.093111, -0.286537, -0.180331, -0.021302, -0.029545, -0.053023, 0.093842, 0.108805, 0.078176, -0.066706, -0.479890, -0.270408, -0.243983, -0.077847, -0.016097, -0.081477, 0.131390, 0.152318, 0.101598, 0.111521, 0.027804, 0.052283, -0.027257, 0.107347, -0.030758, -0.098809, 0.112880, -0.130004, -0.045786, -0.040836, -0.262053, -0.260481, -0.052597, -0.058742, 0.034385, -0.089444, 0.094809, 0.149098, 0.019602, 0.141329, -0.234984, -0.519889, 0.161141, -0.100015, -0.371237, -0.535971, 0.284745, 0.140930, -0.238253, -0.009232, 0.077198, 0.199958, 0.349996, -0.344135, -0.414599, 0.207481, -0.500012, -0.175620, 0.440791, 0.032909, -0.444818, -0.101458, -0.143492, 0.048632, -0.030804, -0.034177, 0.082119, -0.010693, 0.025777, 0.002801, -0.107064, 0.273594, -0.141815, 0.005534, -0.186537, -0.004934, -0.096464, -0.050668, 0.070101, 0.166871, -0.064287, -0.319895, -0.283870, -0.012326, 0.078122, -0.025465, 0.047835, 0.076518, 0.129209, 0.082135, -0.420007, -0.379810, 0.026980, 0.232809, -0.000839, -0.030104, 0.168563, -0.117049, -0.033013, -0.243115, 0.260503, 0.022641, 0.306739, -0.043123, -0.372903, -0.116904, 0.089623, -0.110430, -0.074678, -0.040268, -0.229493, -0.111110, -0.059201, 0.073025, -0.019252, -0.023943, -0.161358, 0.100384, 0.090617, -0.197534, 0.182212, 0.374219, 0.170509, -0.048523, -0.402910, 0.027831, 0.060084, -0.146631, -0.108833, 0.132681, 0.131745, 0.080074, -0.064156, -0.061253, -0.229435, 0.304488, -0.147241, 0.062263, 0.157275, -0.208477, -0.461436, -0.146955, -0.056565, -0.044441, 0.036147, -0.040695, 0.049760, 0.093592, -0.025549, 0.010792, 0.028913, 0.283763, 0.133564, 0.070240, -0.259322, -0.011295, -0.046723, -0.116485, 0.025782, 0.101080, -0.151858, -0.241488, -0.308114, 0.060009, 0.055315, 0.255912, -0.079324, 0.049336, 0.094528, -0.104489, 0.029112, -0.074870, -0.144435, 0.167729, 0.213999, -0.187896, 0.177504, -0.046226, -0.000836, -0.236719, 0.177743, 0.126172, 0.312423, -0.497169, -0.137171, -0.501076, 0.252233, -0.134807, -0.305685, 0.040181, -0.153658, -0.064769, -0.136336, 0.204817, -0.210290, 0.173378, 0.097056, 0.034429, -0.105386, 0.021388, -0.044965, -0.023377, -0.467181, 0.125211, 0.029008, -0.025987, 0.314292, 0.055068, -0.183304, 0.299416, 0.132367, -0.358058, 0.022028, -0.024598, -0.024144, 0.263531, -0.183575, 0.018146, -0.019808, -0.016561, -0.114799, -0.034948, -0.190540, 0.027091, 0.003370, -0.064068, -0.030418, 0.066618, 0.089413, 0.031947, -0.105315, 0.266316, 0.058005, -0.033275, -0.139718, 0.082485, 0.334097, -0.319610, -0.146661, 0.033138, -0.054087, -0.233885, -0.230265, 0.061224, 0.019740, 0.213829, 0.093452, 0.019750, 0.001608, 0.129467, -0.070436, -0.075982, -0.270782, 0.078101, 0.044889, -0.002033, 0.063630, -0.007197, -0.058166, -0.004762, 0.175575, 0.311277, 0.167262, -0.814206, -0.031624, -1.089076, 0.212412, 0.021749, 0.022874, 0.018635, -0.013236, -0.328936, 0.080849, 0.048556, -0.025737, -0.072828, -0.063378, 0.129729, 0.114844, -0.027652, -0.537512, -0.207754, -0.713962, 0.355441, 0.166356, -0.075801, 0.160982, 0.123536, 0.026218, 0.098395, -0.145544, -0.189528, 0.085694, 0.055613, 0.097001, -0.056548, 0.004603, 0.023109, -0.094689, -0.167316, -0.006183, -0.056468, -0.270554, 0.067514, 0.118392, -0.169581, -0.171040, 0.278787, 0.103187, 0.068325, 0.031596, 0.180977, 0.366035, -0.087737, -0.157247, 0.118647, 0.261603, -0.445773, -0.193784, 0.044439, -0.002872, -0.210880, -0.111185, 0.271134, 0.069183, 0.032447, -0.148879, -0.047977, 0.070590, -0.050287, 0.188802, -0.089869, -0.137397, 0.210300, 0.061617, 0.012686, -0.178882, 0.027772, -0.021780, -0.344591, 0.005813, 0.251106, 0.235880, -0.549288, -0.037074, -0.585047, 0.047063, 0.228397, 0.139120, -0.235351, 0.126684, -0.207147, 0.215822, 0.130611, -0.071181, -0.123305, -0.320689, 0.017576, 0.177765, 0.082632, -0.514218, -0.141685, -0.659004, 0.205329, 0.155891, -0.195680, -0.115922, 0.137491, 0.088685, 0.037008, 0.052852, 0.060276, 0.121603, 0.130947, -0.041564, 0.129874, 0.058281, -0.125564, 0.033923, -0.148074, 0.174959, -0.225315, 0.073002, -0.036356, -0.070680, -0.100591, -0.000078, 0.110469, 0.083918, -0.173550, 0.229857, 0.159314, 0.218936, -0.087820, -0.097474, -0.087222, 0.173653, -0.438526, -0.290256, -0.189560, 0.091779, -0.122460, 0.032709, 0.141139, 0.005196, -0.101302, -0.253937, 0.058230, 0.200393, -0.094836, 0.039090, -0.304363, 0.197191, 0.049431, -0.110341, -0.037450, -0.212755, 0.094213, 0.268699, -0.151581, -0.628262, 0.209050, -0.068959, -0.130148, -0.010853, -0.555866, 0.003993, 0.218173, 0.005096, -0.053181, 0.086269, -0.299983, 0.364557, 0.057517, -0.234165, -0.093786, -0.491988, 0.080108, 0.417377, -0.169734, -0.012672, -0.096217, -0.502813, 0.100212, 0.214823, -0.055300, -0.145241, 0.146594, -0.004317, 0.083156, -0.059870, -0.016378, -0.061591, 0.210168, -0.246296, 0.094365, 0.168795, -0.028159, -0.023995, -0.047601, -0.019608, -0.295333, 0.040816, 0.131424, -0.089039, -0.090620, -0.058825, 0.160306, 0.201242, -0.418877, 0.586220, -0.002503, 0.165989, -0.108809, 0.077202, 0.242079, 0.284327, -0.332202, -0.146447, -0.007661, 0.188563, -0.068270, -0.092219, 0.042466, -0.246820, -0.147738, 0.017961, 0.033311, 0.086539, -0.130289, -0.006446, -0.344413, 0.294871, 0.135813, -0.019173, -0.106224, -0.425696, 0.178815, 0.343758, -0.129800, -0.571332, 0.093225, 0.002944, -0.106196, 0.279647, 0.137674, -0.093698, 0.117596, -0.179196, -0.091401, -0.049311, -0.170098, 0.520166, 0.097810, -0.456664, -0.074895, -0.635023, 0.153793, 0.278401, -0.166991, 0.106803, -0.241528, -0.139223, 0.075347, -0.056229, 0.055607, -0.165202, 0.169231, 0.070604, -0.119556, -0.009493, 0.141296, 0.083172, 0.023842, -0.586481, -0.166245, 0.087440, 0.213293, -0.020216, 0.016699, 0.089002, -0.298600, 0.009128, -0.084130, 0.030374, -0.176135, 0.163065, -0.057831, 0.273909, -0.534478, 0.226505, 0.003809, -0.167992, -0.018751, 0.086517, 0.162637, 0.203141, -0.243804, 0.055962, 0.133026, 0.156170, 0.193523, -0.227075, 0.244799, -0.665436, -0.423338, 0.227473, 0.040506, -0.130903, -0.249116, 0.157437, -0.302491, 0.292058, -0.024710, -0.259577, -0.084168, -0.012547, -0.041094, 0.195306, -0.370225, -0.317439, -0.168778, -0.017458, -0.625920, 0.392666, 0.216771, -0.146962, -0.017915, -0.143170, -0.085021, -0.337162, -0.344694, 0.394171, -0.092652, -0.524257, 0.052564, -0.174757, 0.180288, 0.470971, -0.145919, 0.008811, 0.063121, -0.117679, 0.152259, -0.099986, -0.111357, 0.035294, 0.008366, 0.118727, -0.215866, -0.101790, -0.025004, 0.239130, 0.047086, -0.558339, -0.353957, 0.074278, 0.227854, 0.098413, -0.072477, 0.166954, -0.149871, 0.254391, -0.097214, -0.068701, 0.026857, 0.093469, -0.168487, 0.036363, -0.418878, 0.045128, -0.117439, 0.195761, -0.028638, -0.091104, 0.144551, 0.254344, -0.077286, -0.230772, 0.138668, -0.040402, 0.294127, -0.166766, 0.215626, -0.792594, -0.301787, 0.071342, -0.013327, 0.006611, -0.043372, -0.065236, -0.262993, 0.204134, 0.064644, -0.217353, -0.228392, -0.066273, 0.091631, 0.285485, -0.237083, -0.585185, -0.247466, -0.171863, -0.436141, 0.466770, 0.431972, -0.334192, -0.002638, -0.007965, -0.254332, -0.420172, -0.110064, 0.376034, -0.020876, -0.617120, 0.151783, 0.113310, 0.265073, 0.000943, 0.361587, -0.084447, -0.000952, -0.029515, 0.237640, -0.075576, -0.185429, -0.084385, -0.102181, 0.037904, -0.097278, 0.016838, 0.044544, 0.120302, 0.179535, -0.464728, -0.430468, 0.099574, 0.179996, -0.020369, 0.128430, 0.032533, -0.009086, -0.014719, -0.057393, -0.134135, -0.027607, 0.165585, -0.098451, 0.193551, -0.287251, 0.374632, -0.157661, 0.047803, 0.181799, 0.032690, 0.018691, 0.196756, -0.128418, -0.457460, 0.099015, 0.350029, 0.107751, -0.214141, 0.266997, -0.859810, -0.257748, 0.071509, 0.212210, -0.116863, -0.203506, -0.412730, -0.138778, 0.212512, 0.014192, -0.350066, -0.242506, 0.079926
.float 0.370858, 0.152139, 0.026076, -0.471549, -0.477356, -0.230375, -0.157489, 0.428993, 0.367436, -0.420414, 0.054411, 0.141170, 0.057558, -0.259889, 0.226669, 0.470095, -0.029115, -0.522568, -0.129614, 0.013562, 0.243790, -0.170028, 0.165043, 0.009367, 0.063678, -0.174357, 0.006078, 0.009236, -0.122916, -0.188598, -0.039680, 0.000914, 0.076713, 0.024477, 0.092171, 0.077027, 0.150572, -0.252425, -0.368345, 0.075272, 0.030085, -0.026822, 0.089634, -0.024661, -0.159941, 0.162817, 0.001151, -0.313316, -0.024468, -0.075353, 0.014167, 0.188038, -0.299340, 0.319949, 0.096739, -0.257452, 0.020916, 0.271144, 0.070188, 0.009656, -0.185146, -0.274213, 0.155155, -0.138230, 0.067637, -0.230451, -0.061074, -0.742770, -0.052313, -0.097425, 0.435540, -0.064843, -0.178889, -0.015159, 0.028749, 0.069499, 0.004740, -0.264469, 0.026648, 0.051495, 0.008955, 0.238067, 0.034970, -0.160092, -0.260918, -0.380114, 0.150757, 0.418997, 0.228388, -0.450604, -0.136687, 0.016022, -0.115014, 0.016389, 0.084308, 0.028864, 0.118712, -0.618153, -0.192112, -0.143718, 0.458386, -0.220228, 0.183104, -0.107771, -0.019208, 0.058416, 0.068115, -0.554401, 0.039503, -0.140025, -0.078441, 0.095944, 0.115786, -0.127034, 0.164716, -0.070245, -0.194111, -0.039940, -0.387617, -0.206297, 0.176023, 0.023487, 0.079848, -0.093800, -0.070801, 0.274124, -0.459055, 0.094016, -0.346337, -0.064895, 0.436414, -0.110849, -0.134624, 0.283697, 0.333427, -0.381834, -0.080992, 0.168050, -0.148806, 0.134968, -0.202182, -0.186870, -0.017452, -0.161835, 0.300916, -0.202384, -0.317623, -0.179006, -0.119837, -0.259326, 0.226433, -0.042328, -0.212486, -0.292451, -0.209447, -0.719416, -0.029516, 0.333456, -0.182805, 0.122120, 0.101593, 0.082426, 0.112004, -0.151377, -0.036610, -0.550428, 0.140537, 0.194685, 0.004212, -0.141253, -0.179051, 0.001029, -0.024002, -0.068278, -0.045534, -0.228725, -0.103328, -0.113052, 0.010332, -0.053875, 0.191701, -0.236812, 0.282268, 0.026182, 0.011450, 0.144323, -0.481480, -0.293587, -0.008463, -0.170118, -0.101647, 0.202844, 0.038845, -0.023341, -0.142533, 0.402216, 0.184743, -0.479858, -0.339544, 0.169385, -0.469254, -0.485286, -0.112509, -0.271031, -0.098650, -0.246703, 0.159373, -0.218057, -0.010610, -0.024419, 0.047506, 0.166407, 0.096241, 0.329918, 0.076336, -0.089603, 0.152091, -0.304118, 0.003948, 0.098305, 0.044317, -0.215781, 0.044364, -0.002258, -0.378619, -0.385518, 0.177889, -0.029830, 0.072155, 0.071017, 0.124884, 0.083778, -0.263609, -0.376823, 0.020335, -0.668786, 0.366993, -0.150220, -0.315107, -0.123628, 0.013395, 0.104319, -0.076142, 0.015904, 0.170336, -0.097494, 0.026106, -0.017179, -0.289982, 0.086403, -0.095765, -0.128291, -0.012677, -0.094115, -0.205676, -0.170428, -0.056388, -0.243332, 0.007536, 0.202550, 0.049983, 0.007273, -0.098508, 0.038123, 0.214486, -0.134668, 0.037647, -0.001513, -0.438610, 0.365300, -0.379571, -0.859805, -0.012528, 0.037509, 0.136961, 0.534205, -0.549512, -0.344289, -0.345571, 0.139162, -0.306305, 0.046727, -0.013813, -0.296670, -0.176482, 0.036173, 0.180188, -0.198979, 0.121954, 0.034153, 0.182955, 0.059161, -0.058536, -0.026398, 0.329461, 0.037788, 0.010441, -0.280536, -0.114679, -0.020540, -0.145284, -0.017962, 0.108796, -0.257999, -0.102172, -0.291659, 0.019400, 0.018934, 0.077293, -0.017380, 0.056354, 0.140108, 0.034304, -0.748544, 0.094138, -0.396870, 0.155509, -0.118409, 0.122073, 0.088312, -0.080399, -0.001161, -0.517891, 0.058954, 0.330308, 0.285405, -0.363097, -0.068522, -0.473540, -0.007686, 0.012744, -0.133812, -0.029857, 0.015446, -0.157745, 0.033132, 0.214959, -0.123355, -0.002275, -0.033925, -0.163992, -0.065506, -0.401347, 0.063505, 0.157505, 0.250228, -0.490527, 0.250744, -0.178075, -0.086193, -0.009313, -0.056931, 0.328457, 0.237049, -0.094504, 0.003559, -0.075548, -0.364721, 0.414088, 0.022027, -0.267353, -0.117773, -0.087419, -0.144534, -0.090463, -0.112451, 0.158236, -0.068605, -0.106424, -0.116517, 0.185335, 0.177046, -0.090633, 0.051938, 0.432422, 0.345827, -0.029140, -0.134905, -0.122107, -0.125470, -0.136132, -0.212978, 0.098525, -0.124991, -0.107592, -0.350467, 0.168543, 0.077899, 0.137453, -0.173424, 0.061047, 0.095071, -0.076956, -0.227705, -0.049008, -0.412109, 0.328145, -0.209829, 0.087729, -0.018422, 0.224793, 0.016469, -0.417626, 0.036067, 0.393417, 0.170380, -0.481428, 0.132243, -0.503877, 0.098516, 0.056402, -0.253056, 0.004739, -0.051968, 0.143348, -0.037570, 0.151372, -0.009879, -0.154055, -0.079743, -0.040947, -0.234064, -0.143396, -0.105640, -0.088494, -0.183659, 0.090449, 0.231455, -0.111724, -0.179928, 0.142472, 0.023671, 0.204681, -0.113972, -0.314911, 0.061477, 0.045520, -0.177717, 0.054489, 0.151886, -0.173021, -0.083614, -0.184678, -0.390732, 0.003103, -0.341483, 0.217962, 0.078297, 0.079434, -0.144536, 0.050638, 0.166576, 0.189240, -0.042509, 0.398506, 0.089723, 0.003462, -0.130654, -0.040730, 0.182720, -0.077242, -0.412773, 0.098428, -0.094555, -0.248150, -0.161512, 0.117080, 0.089107, 0.157362, 0.010182, 0.079129, -0.002055, -0.097439, -0.190148, 0.000923, -0.275743, 0.165269, -0.063499, 0.058633, -0.075555, -0.066757, 0.132688, -0.504573, 0.305666, 0.163169, 0.139925, -0.234258, 0.011049, -1.033763, 0.114597, 0.046548, 0.204543, 0.084654, 0.019707, 0.063176, -0.082707, 0.178218, 0.017876, 0.098566, -0.285916, 0.009273, -0.028093, 0.064108, -0.463213, -0.187639, -0.531760, 0.024465, 0.036816, -0.082873, -0.129026, 0.246410, 0.037255, 0.098456, 0.164778, -0.322628, 0.214221, 0.126958, -0.007098, 0.086500, 0.185440, -0.031055, -0.086111, -0.106005, -0.147818, 0.080232, -0.184169, 0.144329, 0.016504, -0.102287, -0.522692, 0.219810, 0.183173, -0.060901, -0.003046, 0.223230, 0.146529, -0.117706, -0.094766, -0.054611, 0.502817, -0.352851, -0.299756, 0.013625, 0.021337, -0.222500, -0.143479, 0.103529, 0.111707, -0.017144, -0.121504, 0.053895, 0.158689, -0.073784, 0.178879, -0.132135, -0.100368, 0.312042, -0.144277, 0.122110, -0.254640, -0.119338, -0.030423, -0.266238, -0.066368, 0.344880, 0.109975, -0.152007, -0.087435, -0.835696, -0.058025, 0.100871, 0.229104, 0.025734, 0.187899, 0.112011, 0.186760, 0.251871, -0.005268, -0.208160, -0.650464, 0.114714, -0.157892, 0.167586, -0.913390, -0.294619, -0.663056, 0.058670, 0.196997, -0.137662, -0.012058, 0.295501, 0.077565, 0.172518, 0.095370, -0.225181, -0.101031, -0.099541, -0.018312, 0.082297, 0.073916, -0.060844, -0.057478, -0.185832, -0.024059, -0.037103, 0.036992, 0.136608, -0.124227, -0.067420, -0.373628, 0.127628, 0.433394, -0.204627, 0.508784, 0.061770, 0.404932, -0.198882, -0.216387, 0.091714, 0.177473, -0.393538, -0.198046, -0.079567, 0.097615, -0.224631, 0.153453, 0.056928, 0.059900, -0.036398, -0.240072, 0.149667, 0.056669, -0.165282, 0.052479, -0.128899, 0.145338, 0.355487, -0.057276, 0.046153, -0.424380, -0.021404, 0.236794, -0.060891, -0.868086, 0.177353, -0.023626, -0.088847, -0.117527, -0.148722, -0.108472, 0.033578, 0.207767, -0.058573, 0.019540, 0.066748, 0.303726, 0.008602, -0.136173, -0.133438, -0.587344, 0.095821, 0.113966, 0.052655, -0.280762, -0.102240, -0.494575, 0.188281, 0.141172, -0.093313, -0.228442, 0.278769, 0.045000, 0.157682, -0.024383, -0.050370, 0.198830, -0.300163, -0.112398, -0.017848, -0.117396, 0.154692, -0.182632, -0.189778, 0.044950, -0.332629, 0.042486, 0.218972, -0.106706, -0.022989, -0.444723, 0.054219, 0.410995, -0.629485, 0.372174, 0.012510, 0.404181, -0.241976, 0.052568, 0.361508, -0.052965, -0.381651, 0.062154, -0.066997, 0.212307, -0.019587, -0.074072, -0.215742, -0.163751, -0.208991, 0.096278, 0.040883, -0.040238, -0.327126, 0.030300, -0.207728, 0.272698, 0.320241, -0.240632, -0.079675, -0.439911, 0.061986, 0.367513, 0.049043, -0.923527, 0.290682, -0.020340, -0.132185, 0.208652, 0.201654, -0.428079, 0.055073, -0.196841, -0.254716, -0.259154, -0.169914, 0.210959, -0.151584, -0.190021, 0.104171, -0.165285, 0.095842, 0.269434, 0.112002, 0.006462, 0.001049, -0.257316, -0.182215, 0.223863, 0.042736, -0.154693, 0.026146, -0.138769, -0.101674, 0.039747, 0.039559, 0.180596, -0.095144, -0.582776, -0.215568, -0.056357, 0.088855, 0.055876, -0.172999, 0.141930, -0.263098, 0.198304, 0.099126, -0.190054, -0.012899, -0.000298, 0.102238, 0.012007, -0.435202, -0.037768, -0.015020, 0.161884, -0.071846, -0.039656, 0.290721, -0.245599, -0.096628, 0.170251, 0.272257, 0.148947, 0.129925, -0.300285, 0.177921, -0.183545, -0.304661, 0.335846, -0.094064, -0.181146, -0.272712, 0.024696, -0.145698, 0.210187, 0.090550, -0.223213, 0.016812, -0.067724, 0.040328, 0.263638, -0.328606, -0.619838, -0.093519, 0.196489, -0.455256, 0.351474, 0.469513, -0.088801, -0.287924, -0.169312, -0.250472, -0.350295, -0.049526, 0.151096, -0.131867, -0.434184, 0.036108, 0.000700, 0.358149, 0.145243, -0.079637, -0.021357, -0.062494, -0.141992, 0.118115, 0.098568, 0.042128, 0.097525, -0.118091, 0.083847, 0.025630, 0.003986, 0.043472, 0.181452, 0.143436, -0.572111, -0.434606, -0.030817, 0.294187, -0.030732, -0.031614, 0.110021, -0.190071, 0.130995, 0.212385, -0.197701, -0.069577, -0.181943, -0.106254, 0.018851, -0.039120, -0.093482, -0.014140, -0.124412, -0.145673, -0.222647, -0.097858, -0.022452, 0.154445, 0.110771, 0.259903, 0.115819, 0.202795, -0.472273, 0.398379, -0.313043, -0.301587, 0.145361, -0.246664, -0.022987, -0.224415, -0.324901, -0.084135, 0.161533, -0.026810, -0.366431, 0.072348, 0.160612, 0.096729, 0.239772, -0.390465, -0.506348, -0.279307, 0.304486, -0.240046, 0.281923, 0.277745, -0.151162, 0.014787, -0.170834, -0.074788, -0.502836, 0.013470, 0.220889, 0.287946, -0.553551, -0.075834, -0.016877, 0.272163, -0.267921, 0.092348, -0.038262, -0.144767, -0.054145, 0.250212, -0.015673, -0.026927, 0.052571, -0.129322, -0.185116, 0.136635, -0.057241, 0.099034, 0.118246, -0.090498, -0.541322, -0.331684, 0.026369, 0.225308, -0.049224, -0.020486, -0.080813, 0.014027, -0.041199, -0.113345, -0.138716, -0.079157, 0.183113, -0.010200, 0.140485, -0.088704, 0.010581, 0.125467, -0.182681, 0.062447, -0.270664, 0.030946, 0.068436, 0.064051, -0.014290, 0.166482, 0.252184, 0.131440, -0.490397, 0.296857, -0.036629, -0.258513, 0.006155, -0.138690, -0.073141, 0.042425, -0.084804, -0.067702, 0.071864, 0.073497, -0.263905, 0.014009, 0.177150, -0.065903, 0.043754, -0.176408, -0.227938, -0.495252, 0.090766, -0.219223, 0.307939, 0.269583, 0.204178, 0.098556, -0.393844, 0.055976, -0.099359, 0.106610, -0.108256, 0.210021, -0.487421, -0.027233, 0.004465, 0.324958, -0.221130, 0.143729, -0.100560, -0.011609, 0.052825, -0.022231, 0.011211, 0.001341, 0.025142, -0.193396, -0.124863, 0.098703, -0.015391, 0.054259, -0.046182, 0.039183, -0.462387, -0.265242, -0.145994, 0.207835, 0.104368, 0.150779, -0.185822, 0.154917, 0.067885, 0.044233, -0.043306, -0.113212, -0.299582, -0.207507, 0.106732, -0.358913, 0.166621, 0.227354, -0.152323, -0.036295, 0.091787, -0.063485, 0.065875, -0.335007, -0.166361, 0.222275, -0.218152, 0.151202, -0.623807, -0.309283, -0.522663, -0.020067, 0.001077, 0.062463, 0.045363, -0.109118, -0.103673, -0.058924, 0.147087, 0.005389, -0.214373, 0.114801, 0.134430, -0.002787, 0.058483, -0.225771, -0.058856, -0.236582, 0.157550, 0.051078, 0.458762, -0.045592, -0.089678, -0.049148, -0.342073, 0.006795, -0.283626, 0.272752, -0.260288, -0.002207, -0.310910, -0.063400, -0.110354, 0.391975, -0.202459, 0.173059, -0.048059, -0.129230, -0.017164, 0.180464, -0.101190, 0.057740, -0.029443, -0.235506, 0.023366, 0.060232, -0.204869, 0.015683, -0.221252, -0.086433, -0.192101, 0.094172, -0.161215, 0.140984, 0.047221, 0.075287, -0.202771, -0.134011, -0.202520, -0.204158, -0.136829, 0.121060, -0.086042, 0.061891, 0.218203, -0.385807, 0.242919, 0.475000, -0.123313, -0.051183, 0.087969, -0.118847, 0.027795, -0.508582, -0.287289, 0.062406, -0.093984, 0.302639, -0.077448, -0.201385, -0.050088, -0.184717, -0.363600, 0.033861, -0.311794, -0.005705, -0.395782, 0.138163, -0.205714, -0.306005, -0.227228, 0.049975, -0.060990, 0.338579, -0.423468, -0.183163, 0.133878, -0.030798, -0.067640, 0.207687, 0.268330, -0.303237, 0.116293, -0.109133, -0.304860, 0.333124, -0.383247, 0.355114, -0.070885, -0.012054, -0.433856, 0.047957, -0.097460, -0.015985, -0.470403, 0.133733, -0.192351, 0.137198, -0.112963, -0.143016, -0.148916, 0.099632, -0.127437, -0.015306, -0.049014
.float -0.228981, 0.159532, -0.043633, 0.528708, -0.686128, -0.183688, -0.448724, 0.604467, -0.083507, -0.548267, -0.201146, -0.228407, 0.058126, -0.109320, 0.343509, -0.071673, -0.392441, -0.253111, -0.320061, 0.164385, 0.167635, 0.087119, 0.081762, 0.018470, -0.031347, -0.239309, 0.055038, -0.006849, 0.077322, -0.224139, 0.085117, 0.033788, -0.189403, -0.140035, 0.214614, -0.256344, 0.109955, 0.030782, 0.064967, 0.160228, -0.089719, -0.329736, -0.004865, 0.045248, 0.171554, -0.357939, -0.437912, -0.252191, -0.224535, 0.193963, -0.095379, 0.221015, 0.021004, 0.036783, -0.042925, -0.010397, -0.209040, 0.152630, 0.062624, -0.102702, -0.194058, -0.233950, 0.127917, -0.457908, 0.442057, -0.120729, 0.060962, 0.107177, -0.271371, -0.006308, -0.469963, -0.250707, 0.355616, 0.297807, -0.088766, -0.270426, 0.012341, -0.184405, 0.024468, -0.939419, 0.053393, 0.302345, 0.030784, 0.099457, -0.553127, 0.050051, -0.685331, 0.336629, -0.201991, -0.179347, -0.037016, -0.461238, 0.257591, -0.289929, 0.027144, -0.010231, 0.050546, -0.291254, 0.018055, 0.129686, -0.022971, -0.158665, 0.236095, -0.167201, 0.008750, 0.060694, 0.039456, -0.057417, -0.044739, -0.037399, 0.097387, -0.250544, -0.016077, -0.324221, 0.033216, -0.091947, 0.050595, -0.009562, -0.017471, 0.033690, -0.107410, -0.287059, 0.151479, -0.079600, 0.251486, -0.077358, -0.108499, 0.089335, 0.024335, -0.086689, -0.475422, 0.158383, 0.158246, 0.148887, -0.059092, 0.230491, -0.332478, -0.154312, 0.081903, -0.019018, 0.169644, -0.511497, 0.046971, -0.267621, -0.068538, 0.030138, 0.024390, -0.005687, 0.074081, -0.290464, -0.088560, 0.064548, -0.100608, 0.107183, -0.244605, 0.273869, -0.557334, -0.461061, -0.186376, 0.106401, 0.234685, -0.095662, -0.171322, -0.254484, -0.276534, -0.089766, 0.161650, 0.301767, -0.042185, -0.113115, -0.012981, -0.122872, -0.136094, -0.235017, 0.347507, -0.025757, -0.004580, -0.434151, 0.050400, 0.101485, -0.075295, -0.181212, 0.271278, -0.056024, -0.064589, -0.030106, -0.209604, 0.240232, -0.120279, -0.281472, 0.066250, -0.206686, -0.083480, -0.217484, 0.064362, -0.043424, 0.235316, -0.052589, 0.066813, -0.000725, 0.032485, -0.196157, 0.011028, 0.153855, 0.190677, -0.059016, -0.100265, -0.138980, -0.007274, -0.093789, -0.787557, 0.115423, 0.268869, 0.168715, -0.186920, 0.373221, -0.566351, -0.176354, -0.165156, -0.000841, 0.158400, 0.174663, 0.089106, -0.263347, 0.086217, 0.050814, 0.075727, -0.152848, 0.107809, -0.123047, 0.045164, -0.147958, -0.204581, 0.143670, -0.089567, 0.216671, 0.201449, -0.499337, 0.186267, 0.144189, 0.163906, 0.214603, -0.095714, -0.397882, 0.124818, -0.153547, -0.064512, 0.322473, -0.061709, -0.123596, -0.007680, -0.131843, -0.072217, -0.165993, 0.078606, -0.054740, 0.072298, -0.339835, -0.074996, 0.211760, -0.026847, -0.056489, 0.258077, -0.034996, -0.017442, -0.386016, -0.238188, 0.720865, -0.213689, -0.497314, 0.120522, -0.091172, -0.301575, -0.184237, 0.078231, -0.023671, 0.220210, 0.004884, 0.055841, -0.002758, 0.096534, -0.237699, 0.127793, -0.178127, 0.076816, -0.051611, 0.098917, -0.162514, -0.084708, 0.086207, -0.432411, 0.014234, 0.113553, 0.150044, -0.055314, 0.287944, -0.735283, 0.180618, -0.080388, -0.064212, 0.160110, -0.075592, 0.005468, -0.262770, 0.134246, -0.066863, -0.023439, -0.119689, 0.142600, -0.170333, -0.001963, -0.118756, -0.523951, 0.071399, 0.068547, 0.196497, 0.049188, -0.210436, 0.066190, 0.072589, 0.123597, 0.200448, -0.123374, -0.289912, -0.053879, -0.097119, -0.103003, 0.327750, 0.163142, -0.262558, -0.161663, -0.094878, -0.071058, -0.008326, 0.165883, -0.093000, 0.213991, -0.255708, -0.031349, 0.067616, -0.124093, 0.058302, 0.220913, 0.108307, -0.181079, -0.531222, -0.012347, 0.445596, -0.171949, -0.256393, 0.129119, 0.109671, -0.098188, -0.093885, 0.011097, -0.047027, 0.049972, -0.010396, -0.013045, 0.041240, 0.148804, 0.028223, -0.019881, -0.215396, 0.270347, 0.108570, 0.059488, -0.253065, -0.146162, -0.025335, -0.304733, -0.253124, 0.044611, 0.082721, 0.253726, 0.079755, -0.834143, -0.153196, -0.036555, 0.304091, -0.224278, -0.194597, 0.203597, -0.086318, -0.010822, -0.006440, -0.087654, -0.105766, 0.066887, 0.032532, 0.136013, -0.402057, -0.566538, -0.116324, 0.000438, 0.270384, -0.070325, -0.125009, 0.087522, 0.124054, 0.163879, 0.058570, -0.127056, -0.293965, -0.183233, 0.070117, 0.176082, -0.414737, 0.207531, 0.009904, -0.086951, -0.058024, -0.117935, -0.063852, 0.229786, -0.133691, 0.127921, -0.261507, 0.071099, 0.007253, -0.329634, 0.087890, 0.126427, 0.106528, -0.204439, -0.231592, 0.074588, 0.045016, -0.304945, 0.003428, 0.094687, 0.145475, -0.210490, 0.110576, -0.096625, -0.103593, 0.034754, -0.173573, 0.071400, -0.177060, -0.175980, -0.180837, 0.189400, -0.134681, 0.123369, -0.131918, -0.056443, -0.107151, 0.129569, 0.019181, -0.373080, -0.882230, 0.028080, 0.102731, 0.344576, 0.032549, -0.299021, -0.507143, 0.128507, 0.476844, -0.173231, -0.035237, 0.078732, 0.058056, -0.146588, 0.124949, -0.192413, -0.252281, 0.116369, 0.177239, 0.271657, -0.674888, -0.159079, -0.241284, 0.186577, 0.210265, 0.263321, -0.416504, 0.104285, -0.201485, -0.120874, -0.023862, -0.081050, -0.022901, -0.186012, 0.085755, 0.222727, -0.141848, 0.152877, -0.057314, -0.146905, -0.092725, -0.117608, 0.116482, 0.174978, -0.089959, 0.142278, -0.269315, 0.163093, 0.069483, -0.171850, -0.085362, 0.146037, 0.029155, -0.149688, -0.039472, 0.075303, -0.053013, 0.051117, 0.058453, 0.027146, 0.165045, -0.032819, -0.062228, -0.047463, -0.004261, -0.099921, 0.167961, -0.112285, 0.020413, -0.311969, 0.104011, -0.010299, -0.012353, -0.060903, -0.089951, 0.026870, -0.259389, -0.048812, 0.243809, -0.312622, -0.518581, -0.231955, 0.095908, -0.072755, 0.096504, 0.361999, -0.263144, -0.140634, 0.135682, -0.106280, -0.326300, -0.038355, 0.215336, -0.006292, -0.108260, 0.051068, -0.347966, 0.297866, 0.199629, -0.086151, 0.200526, -0.004590, -0.108267, -0.091748, 0.155125, 0.100130, -0.218163, 0.097257, -0.341374, 0.005504, 0.096327, 0.223365, 0.028637, 0.124078, -0.344362, -0.351771, -0.034363, 0.094745, -0.018097, -0.126288, 0.042325, -0.111020, 0.061071, 0.168584, -0.077383, 0.004914, 0.043039, -0.033820, -0.012398, -0.144754, -0.075296, 0.057941, -0.397642, 0.140333, -0.196303, 0.015027, -0.280388, -0.052934, 0.132931, 0.122305, 0.088847, 0.007669, -0.257814, 0.182017, -0.079144, 0.008712, 0.182097, -0.278124, -0.043897, -0.219956, 0.037471, -0.006194, -0.066800, 0.210478, -0.257871, -0.075172, -0.064635, -0.072495, 0.200896, -0.060075, -0.560569, -0.399500, 0.294577, -0.465252, 0.100144, 0.241517, 0.109175, 0.203664, -0.393310, 0.053047, -0.434826, -0.129467, 0.095125, 0.068740, -0.134164, 0.147435, -0.130234, 0.170610, -0.099213, -0.129335, 0.081819, -0.188531, -0.162360, -0.008201, 0.179315, 0.114425, -0.112287, 0.115005, -0.097860, 0.036391, -0.023322, 0.050712, 0.023786, 0.194523, -0.327375, -0.214180, 0.019738, 0.119245, 0.010943, 0.046415, -0.068743, 0.028486, -0.102010, 0.168191, -0.195745, 0.020391, 0.008719, 0.047145, -0.029184, -0.062301, -0.039614, 0.262988, -0.245538, 0.003026, -0.432243, -0.052994, 0.079147, -0.345687, 0.355028, 0.332644, 0.155881, -0.017061, -0.159809, 0.357725, -0.038531, -0.173413, 0.205518, -0.428252, -0.139126, -0.170081, -0.141287, 0.256127, -0.281995, 0.344292, -0.104969, -0.035924, 0.114695, -0.068194, -0.066676, -0.250595, -0.503222, -0.348464, 0.471712, -0.084773, 0.045671, 0.125152, -0.098224, 0.306484, -0.595781, 0.032778, -0.391837, -0.018647, -0.078848, 0.093516, -0.323552, 0.081606, 0.110802, 0.461955, -0.288819, 0.184921, -0.049824, -0.127926, 0.098471, 0.118955, 0.096946, 0.082539, -0.103274, -0.044381, -0.171704, 0.156269, -0.201265, 0.045188, -0.105688, 0.145535, -0.218294, 0.029637, 0.054628, -0.012564, 0.093060, -0.121934, -0.105277, 0.183194, 0.081955, 0.005289, -0.035344, 0.092737, -0.028388, -0.081729, -0.166036, 0.185551, -0.012424, 0.122031, -0.106579, 0.127563, -0.525257, -0.004213, -0.141126, -0.262718, 0.078357, 0.251658, -0.050855, 0.157972, -0.059058, 0.199346, -0.052423, -0.227220, 0.255542, -0.527098, -0.190745, -0.143815, -0.011263, 0.032206, -0.188983, 0.145141, -0.197783, 0.153552, 0.129185, 0.019746, -0.121121, -0.180986, 0.007744, -0.190767, -0.046091, 0.241983, 0.127980, -0.106250, 0.024360, 0.119274, -0.400310, 0.062236, -0.163723, 0.233254, -0.187451, 0.126932, -0.248922, 0.065622, -0.072935, 0.066218, -0.128519, -0.011737, -0.247661, -0.035388, 0.072330, 0.022979, -0.056789, 0.167345, 0.076681, -0.015013, -0.071665, 0.062412, -0.166901, -0.020248, -0.145096, -0.010456, -0.198757, 0.135322, 0.045093, -0.087095, 0.144880, -0.086119, -0.217878, 0.190318, 0.188539, 0.071253, -0.226062, -0.012727, 0.008273, -0.044317, -0.278477, -0.497829, 0.078330, 0.218071, 0.007004, 0.240161, -0.320035, -0.203872, -0.075306, -0.225590, 0.037842, 0.167946, -0.191478, -0.150501, -0.337062, -0.524747, -0.188776, 0.091203, 0.111423, 0.012779, -0.309228, 0.120366, -0.323788, 0.002893, -0.156094, -0.004196, -0.001576, -0.097536, 0.220537, 0.061887, -0.208312, -0.192439, 0.013626, 0.042092, -0.292399, 0.291318, 0.106215, -0.316322, -0.040363, 0.209703, -0.216756, 0.219721, -0.072178, 0.416069, 0.005754, -0.109652, -0.102723, -0.129325, -0.378238, 0.046663, -0.341205, 0.032783, -0.153697, -0.050253, -0.043242, 0.144756, -0.064943, 0.054707, -0.047584, -0.124968, -0.067220, 0.097520, -0.285262, 0.026523, 0.017269, -0.131979, -0.036270, 0.183018, 0.067478, -0.125571, -0.107205, -0.087378, -0.340221, 0.081440, -0.045223, -0.088853, -0.028670, 0.211323, -0.135101, -0.216974, 0.070331, -0.647113, 0.232046, 0.516212, -0.438109, 0.043094, 0.101473, -0.249543, -0.072575, -0.108990, -0.230408, -0.012460, -0.001086, 0.282800, -0.116527, -0.127613, -0.409154, 0.088849, -0.255841, 0.123263, -0.483781, 0.118142, -0.224335, -0.062862, 0.253278, -0.227571, -0.256368, 0.256976, 0.031783, -0.083797, -0.280899, -0.278715, 0.045126, 0.139701, 0.004656, 0.238059, -0.046735, 0.004212, 0.010222, -0.176471, -0.115274, 0.378992, -0.080856, 0.076515, 0.052698, -0.130128, 0.024323, -0.327731, -0.408318, 0.092512, -0.274581, 0.023634, -0.220643, -0.087958, 0.044380, -0.041143, -0.019720, 0.026123, -0.173049, 0.012883, -0.185010, -0.303793, 0.339862, 0.087798, -0.289532, -0.220234, 0.069512, -0.476837, 0.401125, -0.220320, -0.158280, 0.204027, -0.456562, 0.124517, -0.022476, -0.080867, 0.025902, -0.302651, 0.107611, -0.108084, -0.413014, 0.160448, -0.056305, 0.080250, -0.003122, 0.066225, 0.030551, -0.123895, -0.021228, -0.055581, -0.123623, 0.047481, -0.509717, -0.003525, -0.018790, 0.240433, -0.244519, 0.063120, -0.173380, -0.166028, 0.032884, 0.041124, -0.467115, 0.041700, 0.093302, 0.043486, -0.260726, -0.352049, 0.034827, -0.080466, -0.092006, 0.068773, 0.329052, -0.200810, -0.469042, 0.348766, 0.174963, -0.119895, -0.066026, 0.059633, -0.171812, 0.038141, -0.424743, 0.138646, -0.139461, 0.114821, -0.145561, 0.152231, 0.245562, -0.081823, -0.439905, -0.573514, -0.445651, 0.125283, 0.356243, 0.080701, -0.179527, -0.338656, 0.043729, -0.702038, -0.704537, 0.031986, 0.191951, 0.102036, -0.321648, 0.015466, 0.038065, -0.480851, 0.402094, -0.300930, -0.183267, -0.000735, -0.437657, 0.289191, -0.240737, 0.102906, 0.107618, -0.061341, -0.569703, -0.167795, 0.343627, -0.162732, -0.028478, 0.112490, -0.060739, 0.137053, 0.112535, -0.113594, -0.126632, -0.023659, -0.247176, 0.101625, -0.335301, 0.047609, -0.019743, 0.036345, 0.006363, -0.036226, 0.018771, 0.020713, 0.020333, -0.043873, -0.325990, 0.207641, 0.182364, -0.015296, -0.136792, -0.075119, -0.357000, -0.293945, 0.087091, -0.057445, 0.037001, -0.224103, -0.043230, 0.205388, 0.269846, -0.199751, -0.183321, -0.115223, 0.059966, 0.156569, -0.259035, -0.024243, -0.165479, 0.051625, -0.101011, -0.138201, -0.211232, -0.006350, 0.276829, -0.019361, -0.356117, 0.072306, 0.306208, -0.146100, -0.027579, -0.136185, -0.481233, -0.479287, -0.240722, -0.113500, 0.021384, 0.065727, -0.530973, -0.133255, -0.092274, 0.095365, 0.407351, 0.273360, -0.094845, 0.207098, -0.288939, -0.060287, 0.072056, 0.067428, -0.291191, 0.131842, -0.396853, 0.153054, 0.034651, -0.336987, 0.122130, 0.065834, -0.088739, -0.042389, 0.129842, -0.144963, 0.225792, 0.017727, -0.396506, 0.023148, -0.058895
.float -0.033797, -0.146229, 0.066745, -0.099701, 0.110464, -0.163069, 0.046462, -0.034631, 0.012117, 0.075872, -0.071229, -0.112517, -0.081843, -0.020905, 0.005899, -0.096417, 0.058507, 0.074809, -0.437976, 0.267463, -0.206556, 0.153776, 0.132841, 0.309975, -0.470583, -0.135439, -0.152670, -0.122522, 0.143684, -0.118441, 0.154319, -0.239219, 0.151571, -0.114117, 0.260556, -0.368149, -0.103944, 0.011703, 0.027120, -0.013781, -0.069624, 0.136339, -0.447202, 0.036474, 0.241170, -0.326342, -0.445997, 0.225825, 0.134939, 0.137259, -0.106293, -0.537173, 0.069307, -0.193951, 0.071948, 0.365474, 0.388397, -0.306990, -0.019205, -0.247817, 0.181263, -0.113918, 0.073305, 0.175536, 0.044265, -0.392203, -0.042966, 0.022421, -0.444227, 0.051518, 0.180308, -0.130878, -0.200066, 0.005822, -0.225660, 0.394533, -0.028708, -0.309971, 0.034438, -0.072258, 0.014930, -0.152088, 0.008960, -0.060081, 0.174500, -0.069190, -0.050234, 0.037250, 0.050969, -0.073302, 0.254229, -0.117853, -0.089093, 0.089203, 0.061154, -0.118277, -0.093956, -0.131956, -0.539451, -0.050244, -0.301604, 0.268703, 0.024742, 0.239594, -0.608781, 0.192183, -0.165702, 0.124814, 0.043063, -0.289787, 0.165750, -0.148366, 0.071935, -0.120624, 0.225008, -0.010240, 0.009988, -0.056176, 0.107191, -0.080359, -0.365496, 0.191508, -0.068526, 0.019841, 0.235125, 0.040854, -0.578374, 0.255618, 0.149226, -0.238749, 0.198815, -0.528165, -0.078605, -0.055321, 0.098803, 0.103290, 0.391733, -0.183420, -0.000657, -0.067791, -0.028617, 0.211147, 0.077827, 0.139768, -0.075137, -0.259057, 0.017789, 0.006389, -0.035404, -0.049572, 0.115251, -0.032492, -0.243144, -0.074403, -0.166882, 0.272663, 0.036272, -0.195424, 0.144648, 0.044083, 0.024109, -0.062641, -0.162520, -0.202343, 0.098345, -0.125343, 0.019797, -0.329867, -0.105155, 0.000864, 0.110770, -0.152594, -0.023356, 0.147932, 0.168357, 0.010365, -0.156759, 0.035263, -0.334498, -0.207767, -0.396652, 0.117743, 0.274443, 0.199771, -0.636571, -0.273769, -0.244048, 0.617178, -0.107862, -0.061041, 0.179500, -0.027661, -0.214810, -0.131421, -0.027125, -0.047066, 0.154792, 0.037354, 0.014345, -0.201814, -0.280989, 0.098589, -0.090994, 0.048861, -0.012171, 0.040399, -0.193889, 0.212034, 0.255311, -0.022147, 0.107591, -0.399215, -0.104210, -0.030646, -0.009205, -0.341481, 0.182758, 0.081411, -0.033355, -0.187790, -0.055191, -0.002220, 0.116154, -0.001014, 0.046807, -0.071730, 0.069752, -0.049463, -0.144159, 0.086984, 0.126099, 0.047704, 0.021386, 0.092008, 0.041213, 0.188582, -0.262308, -0.142221, 0.052924, 0.295130, 0.005991, 0.057919, -0.078539, -0.209726, 0.053556, -0.030709, 0.176253, -0.366568, -0.068041, 0.045394, 0.049856, -0.122629, 0.076773, 0.017957, 0.180021, -0.144557, -0.129896, -0.089803, -0.295204, -0.448840, -0.294498, 0.309354, 0.444737, 0.067140, -0.098931, -0.534222, -0.160954, 0.330511, -0.269844, -0.315068, 0.092997, 0.120325, -0.198459, 0.011836, 0.053506, -0.051185, 0.051430, 0.353781, 0.260231, -0.062209, -0.154815, -0.058657, 0.129635, 0.015098, 0.172354, -0.156572, -0.076218, -0.327614, -0.158874, 0.059853, 0.341048, -0.074561, 0.023845, -0.062196, -0.217719, -0.245506, 0.005002, 0.071342, -0.073110, -0.070769, 0.067113, 0.074260, 0.051537, -0.124610, 0.137556, 0.083677, 0.102278, -0.290340, 0.003061, 0.228462, -0.018227, 0.023008, -0.110705, -0.073710, -0.197197, 0.117140, -0.217806, -0.035408, -0.118346, 0.206225, 0.185236, -0.171227, -0.051419, -0.021824, 0.019465, 0.028348, -0.263707, 0.001989, -0.140600, -0.066787, 0.063765, -0.091198, 0.175632, -0.162693, 0.279520, -0.034360, 0.041988, -0.128523, -0.473140, -0.458439, -0.345365, 0.294921, 0.371487, -0.012762, 0.121079, -0.108078, -0.036042, 0.119617, 0.081571, -0.531596, 0.130415, 0.066541, -0.102566, -0.130719, 0.025819, -0.127244, 0.453947, 0.070433, 0.009401, 0.209659, -0.078195, 0.135734, -0.261147, 0.104026, 0.057717, -0.266158, 0.036574, -0.331045, 0.039405, 0.010630, 0.004424, -0.036987, 0.034934, -0.226575, 0.085648, -0.054269, -0.051195, 0.032775, -0.080114, 0.072010, 0.078767, -0.071673, 0.090697, -0.124198, 0.135900, 0.104638, 0.057542, -0.256956, -0.002758, -0.019536, 0.128175, 0.019186, -0.070228, -0.178040, -0.171315, 0.017956, -0.363694, 0.338729, 0.128150, 0.250287, 0.006019, -0.147887, 0.046840, 0.065533, 0.123098, 0.130319, -0.290135, 0.047001, -0.138977, -0.039444, 0.176009, -0.187025, 0.132539, -0.059726, 0.178245, 0.034687, -0.192376, -0.164906, -0.226629, -0.342104, -0.158830, 0.243245, -0.056574, -0.001057, -0.092458, 0.167022, 0.324483, -0.358625, -0.110099, -0.446454, 0.198148, 0.055551, 0.109912, -0.081160, 0.114685, -0.105136, 0.074190, -0.021028, -0.065647, 0.011581, 0.023956, 0.110150, -0.114185, 0.135968, 0.262972, -0.198473, 0.149074, -0.244051, 0.100453, -0.120858, 0.069665, 0.024122, -0.024615, -0.058422, 0.025178, 0.029388, -0.109062, -0.063321, -0.124654, -0.067237, -0.018571, -0.103960, 0.158484, 0.000071, 0.060302, 0.071170, -0.061440, -0.124141, 0.124251, 0.032959, 0.087654, -0.248416, 0.057533, -0.081359, -0.067941, 0.430801, -0.568251, 0.094199, 0.095935, 0.157860, -0.084763, -0.016253, 0.038241, -0.065596, -0.082489, 0.091355, -0.111772, -0.055817, 0.068424, 0.056662, 0.177766, -0.206948, 0.010741, -0.088388, -0.025517, 0.162019, -0.099994, -0.046075, -0.007038, -0.234817, 0.025487, 0.019839, 0.158626, -0.111722, -0.028774, 0.242192, 0.336179, -0.944326, -0.069941, -0.122862, 0.218828, -0.018815, 0.176711, -0.208243, 0.261442, 0.039847, -0.092069, -0.011362, 0.042327, -0.089377, -0.053823, -0.002998, -0.029684, 0.192127, 0.062866, -0.149830, 0.130420, -0.193306, 0.051491, -0.043399, 0.120505, -0.024180, 0.054426, -0.120489, 0.111344, -0.029242, -0.165221, 0.004713, 0.054745, -0.154924, 0.108483, -0.082681, 0.070124, -0.030352, 0.028493, 0.086198, -0.142553, -0.326155, 0.136084, 0.028594, -0.034965, -0.451611, 0.323810, -0.185569, -0.231336, 0.307470, -0.371762, -0.135083, 0.233613, -0.164041, -0.260532, -0.047886, 0.096009, -0.010646, -0.148129, 0.065087, 0.045064, -0.049748, -0.164559, 0.261000, 0.097605, 0.119569, -0.013229, 0.084425, -0.082411, 0.024307, -0.176967, -0.062931, -0.292821, 0.168857, 0.102561, -0.210825, -0.032234, -0.096959, 0.008809, 0.130663, 0.306012, -0.505977, 0.195411, -0.018876, -0.073107, -0.053336, -0.189996, -0.069288, 0.014168, -0.041207, 0.166973, -0.407571, -0.182320, 0.104940, 0.073055, -0.018859, 0.023762, 0.071372, -0.010663, -0.060482, 0.038096, -0.090627, 0.102817, -0.168019, 0.114688, -0.176089, -0.030794, -0.047159, -0.089652, 0.148817, -0.128679, -0.031367, 0.058177, -0.118257, -0.290107, 0.306331, 0.294378, -0.112974, 0.039302, 0.050429, -0.028249, -0.290041, -0.403217, 0.151070, 0.157135, -0.626327, 0.311546, -0.382695, -0.279311, 0.137286, -0.531593, 0.134837, 0.177223, -0.062126, -0.389574, 0.041892, -0.253443, 0.033216, -0.160506, -0.218976, 0.189380, -0.030483, -0.004930, 0.063184, 0.047761, 0.229585, -0.143034, 0.023134, 0.122544, -0.131891, -0.065641, -0.079587, -0.231644, 0.295416, 0.139610, -0.333847, 0.057304, -0.197656, -0.081399, -0.114189, 0.188165, -0.254233, 0.053215, -0.047674, -0.211055, 0.113838, 0.066114, 0.072776, 0.128255, -0.128044, 0.112843, -0.260420, -0.145691, 0.041320, -0.022694, -0.003496, 0.146473, 0.020150, 0.031823, 0.012372, -0.041344, 0.014375, 0.197441, -0.100340, -0.022699, 0.157887, -0.124949, -0.015210, 0.013867, -0.136501, -0.039821, -0.277754, -0.093873, -0.177149, 0.017104, 0.075682, 0.052380, -0.014143, 0.174937, -0.229557, 0.010697, -0.209831, -0.288734, 0.225451, 0.242037, -0.516052, 0.101978, -0.039259, -0.235189, 0.258356, -0.340751, 0.021218, 0.005073, 0.117208, 0.126777, -0.192677, -0.233533, -0.327065, 0.066283, -0.103324, 0.183459, -0.226206, 0.107833, 0.096163, 0.279803, 0.047615, -0.413897, 0.254862, -0.061461, -0.205927, -0.070993, -0.297183, -0.292215, 0.164307, 0.345152, -0.275134, -0.058003, -0.051851, -0.094030, -0.185996, 0.032624, -0.054668, -0.049566, -0.004012, -0.077764, -0.169817, -0.090593, 0.136727, 0.013630, -0.083725, 0.146093, -0.412296, 0.016313, -0.002271, -0.161100, 0.003067, -0.065330, -0.020682, 0.024106, 0.185319, 0.004833, -0.176498, 0.329812, -0.063285, -0.221822, -0.375224, 0.270650, -0.142236, 0.023107, 0.009450, -0.167249, 0.166147, 0.185697, -0.478042, 0.102332, -0.046602, -0.084438, 0.000552, 0.030499, -0.386604, -0.105806, -0.179237, 0.073614, -0.068704, 0.164982, -0.083438, 0.048471, 0.037088, -0.016476, -0.022561, -0.101201, -0.028613, 0.091399, -0.404564, 0.042762, 0.066971, 0.106490, -0.204650, 0.118046, -0.055499, -0.019487, -0.066504, 0.160190, -0.108402, 0.251440, 0.255668, -0.305237, -0.313418, -0.231306, -0.299938, -0.146202, -0.279470, 0.160485, 0.305764, -0.106917, -0.468777, 0.119826, 0.023011, -0.084968, -0.070356, 0.012673, -0.107990, 0.189881, -0.423265, 0.183627, -0.103452, -0.011822, -0.061850, 0.184158, -0.117989, -0.135548, -0.177760, -0.407448, 0.226018, 0.119182, 0.043973, -0.402721, 0.104585, -0.220886, 0.033920, -0.324953, -0.420235, 0.050910, 0.143736, 0.040549, -0.268417, 0.074051, 0.056282, -0.301001, 0.231973, -0.003270, -0.122987, 0.088081, -0.129803, 0.135060, 0.078404, -0.128718, 0.128225, 0.043728, -0.276831, -0.194108, -0.160722, -0.300152, 0.151210, -0.076543, 0.144865, 0.045832, 0.055264, -0.107068, -0.049576, 0.007637, -0.288892, 0.067098, -0.169757, 0.068418, 0.036268, -0.118129, -0.069046, 0.010813, -0.117188, 0.077839, -0.049897, -0.006556, -0.146431, 0.008805, 0.115998, -0.048472, 0.030891, 0.020558, -0.141985, -0.032115, -0.054612, 0.058544, 0.257775, -0.154342, 0.043501, 0.022305, -0.017373, -0.102505, -0.151590, -0.170747, 0.029654, 0.126465, 0.137156, 0.050008, 0.016225, 0.104635, -0.041964, -0.081251, -0.259772, -0.085576, -0.244179, 0.030449, 0.029470, -0.157287, 0.338036, -0.524362, 0.041447, 0.065464, -0.497883, -0.671816, -0.169154, -0.255043, 0.010761, 0.146965, -0.457566, 0.027297, 0.143399, -0.174896, 0.278096, 0.255249, -0.057798, 0.062223, -0.064399, 0.124734, -0.005639, -0.112910, 0.027579, 0.039910, -0.231254, -0.079222, -0.080182, -0.480319, 0.236097, 0.002075, 0.043831, -0.036717, 0.284481, -0.285624, -0.065278, -0.062391, -0.008096, 0.012734, -0.099369, 0.194467, 0.040924, -0.049469, -0.186187, 0.190576, -0.200494, 0.050937, -0.030715, 0.159322, -0.393430, -0.030314, 0.045755, 0.001533, 0.114996, -0.122599, -0.100976, -0.031326, 0.001535, -0.071039, 0.036502, -0.458384, 0.197770, 0.009967, 0.295985, -0.065244, -0.033816, -0.378769, 0.116378, -0.233262, 0.283830, -0.001710, -0.128350, -0.028177, -0.156213, 0.063256, 0.031872, 0.230124, -0.019582, -0.182042, -0.021248, -0.227510, 0.058630, -0.357321, 0.020357, 0.220364, -0.387581, -0.029129, 0.142567, 0.183556, -0.248458, 0.118751, -0.412310, 0.051965, -0.143089, -0.049727, 0.195016, 0.352169, -0.341921, 0.061066, -0.066440, 0.156284, -0.057832, -0.302245, -0.176153, 0.200277, -0.308433, -0.092048, 0.209664, -0.153173, 0.225255, -0.089542, 0.120491, 0.234165, 0.184528, -0.628106, 0.057659, -0.091088, 0.007547, 0.116903, -0.008365, 0.096148, -0.217941, -0.254297, -0.006659, 0.236082, -0.069187, -0.063455, -0.139176, 0.058537, 0.067934, 0.162930, 0.009353, -0.173189, 0.013605, -0.057193, -0.191449, -0.108237, 0.047773, -0.272836, -0.009636, -0.353348, 0.253915, -0.166457, 0.310281, -0.293800, 0.076528, -0.365529, 0.291432, -0.066045, -0.063101, 0.086891, -0.188943, -0.090345, -0.259698, 0.119076, 0.045715, 0.115871, 0.030478, 0.120874, -0.029822, -0.126149, 0.000536, -0.254635, -0.111969, 0.135370, 0.049153, -0.202633, -0.079706, 0.211019, -0.034348, 0.237869, -0.319107, -0.019181, -0.093831, -0.195938, 0.111039, 0.188178, -0.233359, -0.004308, -0.132128, -0.052135, -0.051178, -0.049048, 0.144569, 0.095291, -0.151124, -0.157093, 0.001997, 0.076040, 0.074250, 0.027130, 0.060272, 0.125920, 0.177621, -0.599844, 0.045739, -0.194952, 0.082105, 0.202938, 0.017237, 0.192737, -0.097438, -0.169062, -0.064598, 0.186238, -0.120791, 0.020812, -0.534734, 0.075918, -0.066241, 0.193808, -0.055350, -0.092353, 0.105569, 0.019953, -0.094394, -0.110686, -0.111117, -0.184357, 0.040013, -0.235134, 0.185565, -0.148928, 0.162026, -0.449414, -0.331995, -0.227989, 0.434520, -0.146638, -0.256208, -0.027240, 0.082523
.float-0.290894, -0.072348, 0.167833, -0.009386, 0.129818, -0.009050, 0.215417, 0.033285, -0.101993, 0.086788, -0.203159, -0.086523, 0.115002, 0.000974, -0.230630, -0.039297, 0.139789, 0.052680, 0.155037, -0.330493, -0.040776, -0.058637, -0.180218, -0.115612, 0.060058, 0.012154, -0.084372, -0.156405, 0.091466, 0.037632, -0.003571, 0.063050, 0.125626, -0.040000, 0.041866, -0.207383, -0.152394, 0.341764, 0.130215, 0.075470, -0.004691, -0.014327, -0.175949, -0.137024, -0.188032, -0.153005, 0.091424, 0.266073, -0.008108, -0.038051, -0.077622, -0.175780, 0.103123, -0.037641, 0.065242, -0.294123, -0.059578, -0.078432, 0.108059, -0.101843, -0.024243, -0.031615, 0.229286, -0.148508, 0.095298, -0.108026, -0.291192, 0.072487, -0.112213, 0.180096, 0.180136, 0.042782, -0.217773, -0.153706, -0.179577, 0.099224, -0.006152, -0.355558, 0.172659, 0.172920, -0.209864, -0.067180, 0.069757, -0.108856, 0.156541, 0.017885, 0.239730, 0.038661, -0.047544, 0.001310, -0.134266, -0.117697, 0.099306, 0.077115, -0.088730, -0.304295, 0.210706, -0.040484, 0.131443, -0.264788, -0.006809, -0.015884, 0.089709, 0.058139, -0.154913, -0.072148, -0.051029, 0.061681, 0.049443, 0.188066, 0.142989, -0.176772, 0.085654, -0.059445, 0.035193, -0.199544, 0.017062, 0.149445, 0.086976, 0.110709, -0.166924, 0.055558, -0.222084, -0.093285, -0.073998, -0.110944, 0.058723, 0.243381, 0.031952, -0.146388, 0.072617, 0.120751, -0.262285, 0.119297, -0.079679, -0.171272, -0.060682, 0.031168, 0.146625, -0.157511, -0.115471, -0.074046, 0.054743, 0.046890, -0.072653, -0.008927, -0.119953, -0.270961, -0.109594, 0.157835, 0.083753, 0.038498, -0.013248, -0.188133, 0.053406, 0.014086, 0.096680, -0.433648, 0.154035, 0.189678, -0.070945, -0.109599, -0.081642, -0.168913, -0.026746, 0.010352, -0.011112, 0.215204, -0.062566, 0.027514, -0.227275, -0.199254, 0.080196, -0.229548, 0.064948, -0.239788, 0.159215, -0.182570, 0.072676, 0.004088, -0.036490, -0.056474, 0.267582, -0.010667, 0.021892, -0.036850, -0.122234, 0.075379, 0.139579, -0.047317, 0.117863, 0.062121, 0.138420, -0.054994, -0.135747, -0.256655, 0.195544, -0.545312, 0.323054, 0.082326, -0.300398, 0.032241, -0.139714, -0.085371, -0.084376, -0.040110, -0.109705, 0.113827, -0.165766, 0.024311, -0.064532, -0.013742, 0.108341, 0.057117, 0.014467, 0.047122, -0.157046, 0.039461, 0.085006, -0.051433, -0.026541, 0.051933, 0.143451, 0.050462, -0.046465, 0.045107, -0.004040, -0.297599, 0.145810, 0.052119, -0.028669, -0.173691, 0.100120, -0.017654, 0.209908, -0.317626, -0.015439, -0.291132, 0.288571, -0.045343, -0.019136, 0.014059, -0.006061, -0.253114, 0.149951, -0.158157, 0.020949, 0.034252, 0.123025, -0.078523, -0.075178, -0.039277, 0.155155, -0.393727, 0.142723, -0.055819, 0.103555, -0.138298, 0.091491, 0.100223, -0.049372, -0.157141, 0.123575, -0.018012, -0.021344, -0.074809, -0.063768, 0.006284, 0.089791, 0.002124, -0.113118, 0.039055, 0.089171, -0.064149, -0.044957, -0.041071, 0.133438, -0.013381, 0.136330, -0.166735, 0.139852, -0.019094, -0.187023, 0.404725, -0.291611, -0.058033, -0.051893, -0.096754, -0.454839, 0.018073, -0.125539, 0.020078, 0.068773, 0.050056, 0.076694, 0.049062, -0.178687, -0.013467, 0.150511, 0.080139, -0.057453, -0.034062, 0.144538, -0.021379, -0.155866, 0.024825, 0.114519, 0.205185, 0.256536, -0.157477, 0.036095, -0.159704, -0.003589, 0.231687, 0.064200, -0.778859, 0.059641, -0.030345, -0.040570, -0.061453, -0.154237, -0.118723, 0.094636, -0.021618, 0.017670, 0.188801, -0.066285, 0.103282, 0.160220, 0.016042, -0.213494, -0.022645, -0.027126, -0.275679, 0.077426, -0.103305, 0.026904, 0.015439, -0.127531, 0.082859, -0.005817, -0.076698, 0.077106, 0.046860, -0.027702, -0.042703, -0.090275, 0.020498, 0.089158, 0.171695, -0.109766, 0.048858, 0.049676, -0.096276, 0.034070, -0.278580, 0.026098, -0.024412, 0.196021, -0.316956, 0.027584, -0.215399, 0.010462, 0.142621, -0.413534, 0.141764, 0.141323, -0.443243, -0.214134, 0.107832, -0.249718, -0.063519, -0.046184, -0.051682, 0.151856, 0.177451, -0.091352, -0.069269, 0.028459, 0.119404, 0.070033, 0.069899, 0.029206, 0.181587, -0.018996, -0.299149, -0.057605, 0.423291, 0.428137, -0.433608, -0.051294, -0.287901, 0.098808, 0.016575, -0.050187, -0.652141, -0.127520, 0.009613, -0.159512, 0.146367, -0.381716, -0.006073, 0.137920, 0.021830, -0.030002, 0.058878, -0.159395, 0.119827, 0.124228, -0.088819, -0.120600, -0.041157, -0.083051, -0.114271, 0.017167, -0.051874, -0.062545, -0.009416, 0.032435, 0.163629, 0.056359, -0.091349, 0.044965, -0.039087, -0.060593, -0.217339, -0.154543, -0.001236, 0.162553, -0.021999, -0.163388, 0.035766, 0.159536, -0.049269, -0.000781, -0.233298, 0.035453, 0.083333, 0.400883, -0.300848, -0.214200, -0.180183, 0.096479, 0.219148, -0.866510, 0.085647, 0.046085, -0.193605, -0.013607, 0.097040, -0.094492, 0.092353, -0.269274, -0.819482, 0.181200, 0.163209, 0.165942, -0.134873, -0.028512, 0.027397, -0.001398, 0.048329, 0.036907, 0.135603, -0.057410, -0.130707, -0.196877, 0.204049, 0.363561, -0.476660, -0.180977, -0.144807, 0.155669, 0.019493, -0.272266, -0.258697, -0.108227, 0.000411, -0.160589, 0.106132, -0.241165, -0.068648, 0.067487, -0.146078, 0.208151, 0.025641, -0.290255, -0.115744, 0.189587, -0.011467, 0.063615, -0.029583, -0.175359, -0.155405, 0.005439, -0.005647, -0.051771, -0.134215, -0.066213, 0.078197, -0.054890, -0.054958, -0.006939, -0.057782, 0.215963, -0.254635, -0.105458, 0.098090, -0.012211, 0.144134, -0.183319, 0.090808, 0.016396, -0.178969, 0.100667, -0.070019, 0.104538, 0.216431, 0.402649, -0.465342, -0.082730, -0.150149, 0.053784, 0.111130, -0.256131, 0.056759, -0.156159, 0.175850, -0.226461, -0.280946, -0.248731, -0.028972, 0.150497, -0.275457, -0.083761, 0.120515, 0.019180, -0.098085, 0.072665, -0.053123, -0.105527, 0.115481, -0.111670, -0.056311, -0.000015, -0.063334, -0.283192, 0.272184, 0.369390, -0.111430, -0.126762, 0.069029, -0.020386, -0.264624, -0.345001, -0.037861, -0.171801, -0.046820, -0.109390, 0.254640, -0.204453, -0.067779, 0.037054, -0.417315, 0.142146, 0.044213, -0.003338, 0.004874, -0.137770, 0.110404, -0.219188, 0.078780, -0.037873, -0.187855, -0.043457, -0.044158, -0.070521, -0.255608, 0.421517, -0.188785, -0.057463, 0.086527, 0.088184, 0.186742, -0.713817, -0.425159, 0.051516, -0.035954, 0.102147, -0.071911, -0.150688, -0.239675, -0.143054, 0.072928, -0.004091, -0.281433, -0.290449, 0.080587, 0.029414, -0.050304, -0.251486, 0.042906, -0.159238, -0.008766, 0.072105, -0.108204, 0.099783, -0.232331, 0.104216, 0.077656, -0.101298, -0.163036, 0.018688, -0.148051, 0.047682, -0.106009, 0.025106, -0.141043, 0.248692, -0.030467, -0.081062, 0.053036, -0.189652, -0.116351, -0.059048, -0.416505, 0.057160, 0.038570, -0.156390, 0.095145, 0.144655, 0.018034, 0.304207, -0.004388, -0.170528, -0.229518, -0.014722, -0.063751, 0.416773, -0.025105, -0.108517, -0.139007, -0.098493, 0.049123, -0.192418, -0.124056, -0.293153, -0.150716, -0.050907, 0.510948, -0.074617, 0.074979, -0.095350, -0.108256, -0.381748, -0.243840, -0.022860, 0.271182, 0.545164, -0.165635, -0.149007, 0.197791, -0.170084, -0.187373, -0.901433, -0.131241, -0.093609, -0.116575, 0.058135, 0.106653, -0.298260, -0.049020, -0.191962, -0.096153, 0.093925, -0.096395, -0.393201, 0.050577, 0.091097, 0.028238, -0.029434, 0.078541, -0.231024, 0.041081, 0.088858, -0.020023, 0.006520, -0.177567, -0.000262, 0.070134, -0.070105, -0.000359, -0.083265, -0.093476, 0.016635, -0.001737, -0.008229, -0.626048, 0.208309, 0.166624, -0.360053, -0.024352, -0.102302, 0.011383, 0.023056, -0.342310, -0.171804, 0.301176, -0.097893, 0.217755, 0.054556, 0.126160, 0.106235, -0.151662, -0.390334, 0.035467, -0.137975, 0.096351, 0.090403, -0.018815, -0.184304, 0.120843, 0.165092, -0.034737, 0.009528, -0.162920, 0.051909, 0.070993, -0.374133, 0.440512, -0.606258, 0.002750, -0.142578, -0.276537, -0.475896, -0.171230, -0.022997, 0.236122, 0.139861, -0.353722, 0.149291, 0.036749, -0.231431, -0.096758, -0.305628, 0.128003, -0.078956, -0.162399, -0.033746, 0.080983, -0.078704, 0.045060, 0.182873, -0.298161, 0.075276, 0.050061, -0.661090, 0.200899, -0.136274, 0.026599, 0.098954, 0.144783, -0.297436, -0.050532, -0.095840, 0.091876, 0.025564, -0.074504, 0.144674, 0.119689, -0.046163, -0.056405, -0.047908, -0.251418, -0.032457, -0.034669, 0.082403, 0.025141, 0.147341, -0.107742, -0.250196, 0.139995, 0.095184, -0.183697, 0.000785, -0.166406, -0.091855, 0.364767, -0.221566, 0.030687, 0.019496, 0.266337, -0.037176, -0.148629, -0.305571, 0.129920, -0.068940, 0.056333, -0.022944, -0.235735, 0.033024, 0.134754, 0.154674, -0.028592, 0.223221, 0.093731, -0.037339, -0.256630, -0.246575, 0.072129, -0.716516, 0.049756, -0.317594, -0.435422, 0.250485, 0.206684, -0.036763, 0.312379, -0.007108, -0.168091, 0.265368, -0.129266, -0.051392, 0.162925, -0.188719, -0.256674, -0.034876, -0.181547, 0.113604, 0.025238, -0.110957, 0.054955, -0.011202, -0.151610, -0.067397, -0.006735, -0.620153, 0.015431, -0.159223, 0.227303, 0.162855, 0.250504, -0.798622, -0.034628, -0.353594, 0.268099, 0.051189, -0.010538, 0.168715, -0.001327, -0.152294, -0.012987, 0.031983, -0.213582, 0.004277, -0.108613, 0.115418, 0.032775, 0.071250, 0.135891, -0.290230, 0.118469, 0.069764, -0.117738, -0.075198, -0.330286, 0.087400, 0.065716, -0.180805, 0.170764, -0.194717, 0.095939, -0.162356, -0.213240, -0.270276, 0.279251, 0.077274, -0.167740, 0.010849, -0.033341, 0.016255, 0.054769, -0.094545, 0.111835, 0.020419, 0.009351, 0.216581, -0.171838, -0.063385, 0.028129, -0.772883, -0.080939, 0.145515, -0.496901, 0.146976, 0.113136, 0.198506, 0.142256, 0.094252, -0.410837, 0.026268, -0.084042, -0.217822, 0.104122, -0.245118, 0.049851, 0.113013, 0.009871, 0.037911, -0.065593, -0.321725, 0.049102, 0.198630, -0.128748, -0.038840, -0.098454, -0.290043, 0.208836, -0.143678, 0.213642, -0.150415, 0.228876, -0.481709, -0.092717, -0.505290, 0.268773, 0.216419, 0.003284, 0.115772, -0.126118, -0.201049, -0.125115, 0.199487, 0.035861, -0.010446, -0.206757, -0.070082, -0.153308, 0.280039, 0.052068, -0.246780, 0.198787, 0.221536, -0.152790, -0.048484, -0.135609, 0.010942, -0.241749, -0.117748, 0.245920, -0.151220, 0.003229, 0.027762, -0.481434, -0.338889, 0.274848, 0.210527, -0.211629, 0.050546, -0.055894, -0.198350, 0.027204, 0.107845, -0.062969, 0.074663, -0.113345, 0.190586, -0.290866, -0.115962, 0.022353, -0.584049, -0.128750, 0.212023, -0.315775, 0.257208, -0.084329, 0.253959, -0.243437, -0.057517, -0.248683, 0.056466, -0.177436, -0.085275, 0.188123, -0.081174, 0.145114, 0.212973, -0.107869, 0.169721, 0.090140, -0.188967, 0.070839, 0.297624, -0.265326, -0.150726, -0.204529, -0.241056, 0.103610, 0.089449, 0.086331, -0.219896, 0.164494, -0.033281, -0.145099, -0.358767, 0.155500, 0.030850, 0.183346, -0.210081, -0.143442, -0.036908, 0.006267, 0.093599, 0.082070, -0.143841, -0.001034, 0.120009, 0.010259, 0.149623, 0.051246, -0.165873, 0.222948, -0.118437, -0.063526, 0.046942, -0.233679, 0.220244, -0.374757, 0.031225, 0.075945, -0.160449, 0.032941, 0.026135, -0.396043, 0.143724, -0.057372, 0.170478, -0.442440, -0.012201, 0.194472, -0.110953, 0.010809, -0.061012, -0.188351, 0.112266, -0.045427, 0.194378, 0.029485, -0.169869, -0.217892, 0.039671, -0.000755, 0.101211, 0.014608, 0.215651, -0.035825, 0.129295, -0.255489, -0.068396, -0.040328, 0.013113, 0.002273, 0.019558, 0.056077, 0.088101, 0.097252, -0.009617, 0.049321, 0.102023, -0.034075, -0.103015, 0.023886, 0.249852, -0.082639, -0.093932, -0.153877, -0.007110, -0.498804, 0.053845, 0.111751, -0.283695, 0.191629, 0.146309, -0.094742, -0.272112, 0.026943, -0.207329, 0.260434, -0.289043, 0.005196, 0.129917, 0.135843, -0.409327, 0.147929, -0.094393, 0.149446, 0.121118, -0.015768, 0.217072, -0.051576, -0.058101, -0.026413, 0.064823, -0.027340, -0.033050, -0.075136, 0.164529, -0.331699, 0.198569, 0.075494, -0.376182, -0.140316, 0.317522, -0.293524, 0.100829, -0.178000, 0.000571, -0.453220, 0.082720, 0.132624, -0.076225, -0.044340, 0.017310, -0.165344, 0.075876, 0.027296, -0.088540, 0.216388, -0.041568, -0.175538, -0.104242, -0.105918, 0.057626, 0.044065, 0.097505, -0.000132, -0.046199, -0.045620, 0.092126, -0.016406, 0.035969, 0.085351, 0.266920, -0.054608, -0.014849, -0.119768, -0.057545, 0.139674, 0.142089, 0.000241, 0.068905, 0.036943
.float 0.188731, 0.035619, -0.150660, -0.142350, 0.098940, -0.323455, 0.047889, 0.136579, -0.612329, 0.100759, 0.107949, -0.079903, -0.224752, -0.027103, -0.276543, -0.045430, -0.131617, 0.112178, -0.073699, 0.068296, -0.079594, 0.098262, -0.069493, 0.050116, 0.129239, 0.118941, 0.251697, -0.050733, -0.236701, 0.107745, -0.055001, -0.282625, 0.090760, -0.318690, 0.044980, 0.050778, 0.368248, 0.039926, -0.264041, -0.028008, 0.276000, -0.165074, -0.181572, -0.466655, -0.099060, -0.213496, 0.179224, 0.050557, -0.356021, 0.134850, 0.003494, -0.107502, 0.075592, 0.015560, -0.289529, 0.001672, 0.170450, -0.104802, 0.016064, 0.003811, -0.070631, -0.068920, 0.050898, 0.023660, -0.022083, -0.088357, -0.013999, 0.135345, -0.031553, 0.082495, 0.016492, -0.033354, 0.126581, -0.020194, -0.078001, 0.078624, 0.121861, 0.049120, -0.142515, -0.025349, -0.069578, -0.046602, -0.168642, -0.066768, 0.103388, -0.083858, 0.042552, 0.065116, -0.667732, 0.164503, 0.148309, -0.175183, -0.156218, 0.001716, -0.259633, -0.230287, -0.390724, 0.199156, 0.086769, 0.003512, 0.101529, -0.034965, -0.136575, 0.262233, 0.106523, 0.036100, 0.079454, -0.121291, -0.182549, 0.104812, 0.049295, -0.223942, 0.067833, 0.092773, 0.125739, 0.028596, 0.371794, -0.296959, 0.184239, -0.115774, 0.069174, -0.015972, -0.255014, -0.452516, -0.085379, -0.233711, 0.134348, 0.128789, -0.365561, 0.084635, 0.057303, -0.116131, 0.170785, -0.172101, -0.413781, -0.041915, 0.306438, -0.029618, 0.096357, -0.124009, -0.092149, -0.119220, 0.065833, -0.057818, 0.085004, 0.076916, -0.037019, -0.016176, -0.264119, 0.101487, -0.020132, -0.134617, 0.078632, -0.017429, -0.066906, 0.126337, 0.109191, 0.142484, -0.088562, 0.162536, -0.053795, -0.269772, -0.175326, -0.109017, 0.297832, -0.071848, 0.128284, 0.053146, -0.275565, 0.057363, -0.453800, -0.169571, -0.115476, -0.152988, -0.060530, -0.223401, -0.034615, 0.056797, -0.069585, 0.066956, -0.023030, -0.226561, 0.058630, 0.202521, -0.033393, -0.053691, -0.087753, 0.203562, -0.044909, 0.078123, -0.028191, 0.010222, -0.038817, 0.066021, 0.068944, 0.310112, 0.569678, -0.332424, 0.135320, -0.346773, -0.132859, -0.106652, -0.303256, -0.394769, -0.130829, -0.532977, -0.029712, 0.094142, -0.278131, 0.285811, 0.186330, -0.263031, 0.102594, -0.096423, -0.291815, 0.183006, 0.257699, -0.035996, 0.058383, -0.128678, -0.005440, -0.159424, 0.086044, -0.054726, -0.098560, -0.065046, -0.158485, 0.088766, 0.032879, 0.053492, 0.119457, -0.174444, 0.127655, -0.102606, -0.130650, 0.109715, 0.099541, 0.046019, -0.132647, 0.105889, -0.102802, -0.099778, -0.052592, -0.096426, 0.238374, 0.017215, 0.637644, -0.352500, 0.134904, -0.064628, -0.038745, -0.049617, -0.327572, -0.310988, -0.407033, 0.259927, 0.067560, 0.089957, -0.188359, 0.351496, -0.423209, -0.636584, -0.031806, 0.145333, -0.020713, -0.013794, -0.150662, 0.045723, -0.272655, 0.067868, 0.120313, -0.340935, -0.024573, 0.161725, 0.000686, 0.285079, 0.376815, -0.704064, 0.144750, -0.329136, -0.099436, 0.064736, -0.334973, -0.085129, -0.016782, -0.261084, -0.368964, 0.224357, -0.017705, 0.214197, 0.175395, -0.401828, 0.165548, -0.132631, -0.314772, -0.077812, 0.071235, 0.002042, 0.181447, 0.054268, -0.158218, -0.264493, -0.025698, 0.017230, -0.025803, 0.023457, -0.210530, 0.149515, -0.166102, -0.021519, 0.014934, -0.323426, 0.070379, 0.028924, -0.090030, 0.092017, -0.093860, 0.104318, -0.254678, 0.118320, -0.089844, -0.099646, -0.022553, 0.041763, 0.274199, 0.395789, -0.020178, -0.244073, 0.067748, -0.124190, -0.045074, 0.118386, -0.138424, -0.138363, -0.305614, 0.194162, -0.157975, -0.322563, -0.337913, 0.375898, -0.231846, -0.252711, -0.207406, 0.137625, 0.304925, -0.048253, -0.173984, 0.117415, -0.143856, -0.067880, -0.308746, -0.520736, 0.047021, -0.024633, -0.050688, -0.096090, 0.351174, -0.401540, 0.076903, -0.439117, -0.059716, 0.022627, -0.038685, 0.113193, 0.111103, -0.527639, 0.139690, -0.149024, 0.102596, 0.070504, -0.297800, -0.298011, 0.245794, -0.172833, -0.252740, -0.032877, -0.052439, 0.134376, -0.169629, 0.249193, -0.207629, -0.369932, -0.078311, 0.047659, 0.056399, 0.022203, 0.454984, 0.003530, -0.167252, -0.397365, -0.154322, -0.689713, -0.488932, -0.250828, -0.263865, -0.048576, -0.216516, -0.001162, -0.376036, 0.233449, -0.100175, 0.136235, 0.064225, -0.102154, 0.167454, -0.002767, -0.364767, 0.206360, 0.069949, 0.127404, -0.029416, 0.160977, -0.214996, -0.122649, 0.065777, -0.097169, 0.051326, 0.100837, -0.098000, -0.181773, -0.180977, 0.107771, -0.201600, 0.001184, 0.106372, -0.198173, 0.125819, -0.044626, -0.191821, -0.012606, -0.083465, 0.229108, -0.571356, -0.128119, 0.059535, 0.181896, 0.048919, -0.085216, 0.042992, -0.131750, 0.131974, 0.120326, -0.260882, 0.083884, -0.303375, -0.155033, 0.048193, -0.043675, -0.464699, 0.186115, -0.191123, 0.092164, 0.056803, 0.025386, -0.122912, -0.049728, -0.321927, -0.163425, 0.167657, 0.110081, 0.027861, 0.102479, -0.277085, -0.219684, -0.124630, 0.185265, 0.119613, -0.137764, 0.025955, 0.117400, 0.007096, -0.321355, -0.162991, -0.035908, 0.143823, 0.039297, -0.038457, 0.225791, -0.248396, 0.021103, -0.542330, 0.084061, -0.020335, -0.007321, -0.098891, -0.189513, -0.318557, 0.365447, 0.109583, -0.016710, -0.103987, 0.217003, -0.396879, 0.046277, -0.004482, -0.133684, -0.079282, -0.005537, -0.131728, -0.128976, -0.032849, 0.202546, 0.106718, 0.026377, 0.145500, -0.065472, -0.171462, 0.096042, -0.383666, -0.028404, -0.477260, 0.162998, 0.082081, 0.060257, 0.013289, 0.193687, -0.280922, 0.207901, 0.166522, -0.052820, 0.257941, -0.202425, -0.094882, 0.017961, 0.071582, 0.095755, -0.065416, 0.006819, -0.277610, 0.019880, -0.161073, 0.016399, 0.027990, 0.077171, -0.056544, -0.291025, -0.255929, 0.348562, 0.218441, 0.121244, -0.047265, -0.451207, -0.316337, -0.241806, -0.168494, 0.179757, 0.065408, -0.059306, -0.034052, 0.081321, -0.134593, 0.069229, -0.118698, -0.015116, 0.031200, -0.056544, -0.064691, 0.046085, -0.131583, -0.023062, -0.284517, -0.097086, 0.302062, 0.061252, -0.663429, 0.031646, -0.462637, 0.276010, -0.010789, 0.183800, -0.176411, 0.140382, -0.283629, 0.101443, -0.040768, -0.080816, 0.190497, 0.023047, -0.099582, -0.062090, -0.099330, -0.121183, -0.021105, -0.126527, 0.136506, 0.004899, 0.022161, 0.080137, -0.336883, -0.055458, -0.330329, -0.257590, 0.222739, -0.032635, -0.188888, 0.028615, -0.208210, 0.216851, 0.078277, 0.147217, 0.045808, -0.349879, -0.111356, 0.067359, -0.161094, -0.053846, -0.070865, -0.164088, 0.028085, 0.138018, -0.131902, 0.034601, 0.183580, -0.037892, 0.352629, -0.601167, -0.497800, 0.438160, -0.420032, 0.211239, -0.186885, -0.556277, 0.292350, -0.301470, -0.049698, 0.000965, -0.049547, -0.213430, 0.091908, -0.218592, -0.012437, 0.380447, -0.130916, 0.199589, 0.091720, -0.158726, -0.014454, 0.187192, -0.227516, 0.111076, -0.114704, -0.165769, 0.025032, -0.097436, -0.976151, 0.117469, -0.271522, 0.216520, -0.184100, 0.289903, -0.242037, -0.043788, -0.437480, 0.153341, -0.139938, 0.071841, 0.215919, 0.002800, -0.019652, -0.103439, 0.126401, -0.069087, -0.054284, 0.043480, 0.065567, 0.120879, -0.054160, 0.088591, -0.154906, 0.057606, -0.240944, 0.000881, 0.049831, 0.088733, -0.084007, -0.077995, -0.122619, 0.132079, -0.313810, 0.150039, 0.131016, -0.569082, 0.068635, 0.168402, 0.014527, 0.119873, -0.057759, 0.036009, -0.123950, 0.210174, -0.210024, 0.201102, -0.069661, -0.037887, 0.279059, -0.104235, -0.134829, 0.089183, 0.010722, 0.146338, 0.048912, -0.795128, 0.214113, -0.435465, 0.000565, -0.125004, -0.052473, -0.258341, 0.254630, 0.079315, 0.042814, 0.233114, -0.071128, 0.005615, 0.067197, -0.191962, -0.020766, 0.051185, -0.269456, -0.051902, 0.130408, -0.240195, 0.115295, -0.046883, -0.421518, 0.019343, -0.263481, 0.403747, -0.193645, 0.273453, -0.652650, -0.156912, -0.640276, 0.242358, 0.089650, 0.038659, 0.059461, -0.206100, -0.071958, -0.123398, 0.093654, 0.080576, 0.039075, -0.044006, -0.141921, 0.228013, -0.132214, 0.169286, -0.337112, 0.151040, 0.083651, -0.077920, 0.209993, -0.140936, -0.162453, -0.253867, -0.093509, 0.033142, -0.386114, 0.084878, 0.200628, -0.581349, 0.341554, -0.014524, -0.092440, -0.019818, -0.063099, 0.129353, -0.100271, 0.179107, 0.027438, -0.114294, -0.123582, -0.063794, 0.258892, 0.399757, -0.172434, -0.435589, 0.225667, 0.017958, 0.099700, -0.203188, 0.207349, -0.122838, -0.038477, -0.281598, -0.149535, 0.011277, 0.198446, -0.044997, 0.085379, 0.220910, 0.064499, 0.059573, 0.086834, -0.063298, 0.122413, -0.042361, -0.347375, -0.013316, 0.138369, -0.165868, -0.026523, -0.070378, -0.154911, -0.138722, 0.018676, 0.315164, -0.017576, 0.169723, -0.819348, -0.375701, -0.594999, 0.145220, -0.092643, 0.090404, -0.152438, -0.148621, 0.057110, -0.106870, 0.009070, 0.105605, -0.046957, 0.104218, -0.035593, -0.084458, -0.089424, 0.202186, -0.425942, 0.267908, 0.133444, 0.120221, -0.046560, -0.080552, 0.086448, -0.189612, -0.130218, 0.084700, -0.366930, 0.133050, 0.265461, -0.409314, 0.230802, -0.435430, 0.154246, -0.309392, 0.144952, 0.167380, -0.123822, -0.186928, 0.205088, -0.021091, 0.059552, -0.024117, -0.075891, 0.510424, -0.400419, -0.420730, 0.504849, -0.134322, -0.050727, 0.282049, 0.157547, -0.188995, -0.152752, -0.106349, -0.165248, 0.072825, -0.068387, 0.021047, 0.147296, 0.078971, 0.143407, 0.028275, 0.194972, -0.067063, 0.220135, -0.057647, -0.391056, -0.025601, 0.224660, -0.212781, 0.097051, -0.102467, 0.149859, 0.054927, -0.122830, 0.310357, -0.270792, 0.168921, 0.015817, -0.196017, -0.231219, -0.143943, -0.670982, -0.076819, -0.316212, -0.076926, 0.074134, -0.026716, -0.228146, 0.277449, -0.162015, 0.172981, 0.093065, -0.303718, -0.070506, 0.246999, -0.325862, 0.305592, -0.116262, 0.025113, 0.105010, -0.251276, 0.265847, 0.112726, 0.071676, -0.208566, -0.181650, -0.012806, 0.383094, -0.543993, 0.034400, -0.387839, 0.139203, -0.476729, -0.014117, 0.134185, -0.164019, 0.079193, 0.124894, -0.304184, 0.377602, -0.250137, -0.540251, 0.163870, -0.101986, -0.237353, 0.311501, -0.130387, -0.116441, 0.202240, -0.046966, 0.153124, -0.040833, -0.105103, -0.228236, 0.082292, -0.174358, 0.103522, -0.035992, 0.001062, 0.142172, -0.064923, 0.125238, -0.147663, 0.164787, -0.025274, -0.098908, 0.012614, 0.147244, -0.070772, 0.022385, -0.175839, 0.278324, -0.009875, -0.037908, 0.317399, -0.484482, 0.068875, 0.172297, -0.327522, -0.201492, -0.040124, -0.795081, -0.086361, -0.241151, 0.146540, 0.206636, 0.050489, -0.393535, 0.219234, -0.113836, 0.091384, 0.008546, -0.142788, 0.124054, 0.012533, -0.149853, 0.084493, -0.094103, 0.019219, 0.073735, -0.197088, -0.245813, 0.180036, 0.327001, -0.338758, 0.019885, -0.050580, 0.126399, -0.467443, -0.125622, -0.167580, -0.143293, -0.404094, -0.178500, 0.259573, -0.318484, 0.140992, 0.231809, -0.316027, 0.358654, -0.218771, -0.590205, 0.006533, 0.216387, -0.338631, 0.267288, -0.309014, -0.075884, 0.003720, -0.098073, 0.076354, -0.070661, -0.240007, 0.020256, 0.080968, -0.137726, 0.220335, -0.024983, -0.200038, 0.116417, -0.047039, -0.100012, -0.116376, 0.099088, -0.124930, -0.178589, 0.038514, -0.013984, -0.090924, 0.040243, 0.211770, 0.921631, 0.062468, 0.245955, -0.022836, -0.139664, 0.177165, 0.069139, -0.525903, -0.284889, -0.067870, -0.666701, 0.184724, -0.103864, 0.070223, 0.249496, -0.088809, -0.221342, 0.136642, -0.179047, 0.037255, -0.102373, -0.117658, -0.175958, 0.061947, -0.472224, 0.179565, 0.076657, -0.017567, 0.135525, -0.042075, -0.236903, 0.348224, 0.564735, -0.458754, 0.210434, -0.296684, -0.136774, -0.243376, -0.303225, -0.062518, 0.000254, -0.038880, -0.086283, 0.145816, -0.102222, 0.243195, 0.175859, -0.698116, 0.028934, -0.216494, -0.506366, 0.036608, 0.348029, -0.193838, 0.147854, -0.277558, -0.203557, -0.083107, -0.131296, 0.031702, -0.240455, -0.155780, -0.202727, 0.210256, -0.012014, 0.080020, 0.151188, -0.182257, 0.060721, -0.196144, -0.094087, 0.053880, 0.263289, -0.174460, 0.161381, -0.172735, -0.076839, -0.051363, 0.141248, -0.244217, 0.330479, 0.308538, 0.314322, -0.312543, 0.044799, 0.270689, 0.081548, -0.275272, -0.275668, -0.290568, -0.551228, -0.119018, 0.091724, -0.026043, 0.276837, 0.118596, -0.413732, -0.102667, -0.281444, 0.201704, 0.227888, -0.039138, -0.282364, 0.081480, -0.118561, 0.132616, 0.185866, -0.007942
.float 0.130703, -0.076092, -0.170999, 0.290019, 0.270187, -0.322901, 0.230383, -0.420586, -0.270786, 0.057248, -0.415386, -0.013669, 0.008134, -0.409863, -0.136066, 0.289674, -0.031062, 0.182456, -0.200709, -0.399351, 0.207499, -0.132586, -0.372905, 0.176962, 0.233909, -0.144852, 0.059959, -0.206833, -0.100678, -0.066518, -0.021348, 0.085320, -0.103876, -0.019462, -0.240079, -0.051990, 0.037364, 0.032433, 0.075521, -0.171958, 0.164248, 0.141574, -0.260596, -0.045469, 0.124090, -0.150661, 0.121382, -0.004467, -0.221933, -0.232270, 0.122616, 0.158196, 0.332618, 0.254758, 0.125039, -0.208753, 0.293592, -0.058073, -0.018123, -0.186339, -0.093435, -0.301381, -0.451795, 0.084907, 0.125690, -0.058570, -0.081174, 0.409039, -0.560413, -0.548176, -0.193738, 0.199377, 0.163439, -0.308935, -0.182035, 0.032602, 0.058538, 0.157459, -0.132428, -0.460910, 0.113887, -0.186167, -0.314441, 0.325387, 0.264631, -0.406355, -0.002684, -0.251934, -0.127719, 0.086117, -0.423602, 0.230435, -0.315912, 0.038032, -0.205376, 0.108204, -0.063539, 0.198874, -0.271928, 0.044032, 0.152482, -0.239815, -0.495859, 0.118281, -0.036767, -0.242419, 0.084630, -0.094163, 0.221896, -0.123154, 0.084670, 0.159185, -0.129255, -0.412741, -0.083190, -0.020023, 0.125539, 0.284602, -0.427077, -0.527147, -0.055544, -0.070804, -0.105113, -0.086461, -0.166989, -0.003873, -0.075048, 0.249660, -0.264796, -0.170901, 0.135734, 0.100428, 0.138453, 0.227395, 0.107760, -0.275628, 0.142036, 0.011924, 0.088601, 0.010339, -0.202764, -0.094465, -0.286177, 0.202944, -0.209683, -0.466076, -0.291517, 0.261934, -0.246109, -0.403962, -0.339968, 0.284000, 0.057900, -0.266462, -0.073687, -0.245965, -0.279756, 0.276874, -0.388694, -0.252745, -0.047563, 0.228358, 0.138026, 0.452509, 0.099434, -0.416263, -0.119962, -0.014779, 0.005700, -0.006783, -0.353239, 0.219178, -0.274223, -0.098656, 0.169434, 0.039937, -0.084592, 0.439633, -0.302655, -0.511476, -0.450459, 0.115619, -0.249023, -0.490316, -0.038734, 0.075034, 0.023452, 0.064884, -0.293628, -0.542474, 0.130638, 0.143617, 0.002629, -0.056568, 0.135121, -0.037444, -0.503891, 0.033986, -0.359733, -0.351356, 0.120665, -0.245762, -0.096368, -0.027941, -0.405543, 0.227182, 0.076052, -0.202187, 0.012151, 0.293832, -0.508938, -0.154871, 0.253152, 0.354448, -0.089942, -0.123219, 0.119008, -0.151159, -0.052813, -0.123664, -0.251026, 0.065706, -0.255206, -0.054607, -0.056577, 0.096900, -0.111894, 0.112374, -0.471987, 0.058823, 0.003100, 0.029969, -0.154397, -0.125950, -0.279665, 0.230823, -0.016905, -0.019955, 0.042968, -0.017144, -0.315389, -0.105081, 0.077294, 0.151880, -0.000734, -0.066090, -0.306089, -0.070884, -0.072880, -0.035556, 0.030648, -0.003334, 0.113401, -0.093236, -0.352855, 0.187705, 0.102425, -0.152087, -0.084849, 0.223374, -0.454569, -0.092084, -0.000611, -0.025620, 0.011844, 0.031391, 0.023094, -0.076242, 0.040529, -0.063939, -0.083435, 0.042099, 0.023431, -0.025723, -0.103558, 0.001397, -0.490650, -0.057679, -0.769584, 0.102541, 0.108576, -0.071385, -0.502257, -0.208501, -0.154058, 0.344695, 0.192106, -0.160404, -0.179540, -0.014544, -0.137445, -0.030763, 0.063928, 0.209509, 0.051805, 0.156738, 0.024804, -0.158909, -0.058892, -0.321471, -0.256355, 0.266026, 0.020010, -0.185739, -0.028553, 0.086106, -0.075187, -0.291843, -0.672902, 0.057502, -0.020309, 0.067389, 0.084621, -0.242728, -0.538112, 0.276047, -0.095101, 0.011130, -0.168185, 0.340242, -0.277467, 0.158818, 0.213673, 0.040656, 0.166310, -0.019444, -0.313761, -0.045290, -0.209817, -0.122266, 0.123450, -0.085072, 0.049454, -0.308977, -0.226318, 0.172495, 0.172668, -0.261638, -0.189553, 0.166106, -0.379765, 0.064355, 0.016947, -0.007098, -0.241918, -0.292478, 0.219615, 0.097178, -0.030931, -0.450197, -0.217093, 0.385974, 0.152486, -0.093908, -0.163797, 0.110024, 0.167682, -0.217641, -0.131827, 0.135393, -0.135722, 0.099907, -0.155736, 0.006324, -0.251757, 0.359060, -0.446870, -0.110701, -0.365505, 0.267910, 0.042594, -0.115487, 0.107532, 0.340642, -0.473808, 0.182608, -0.289910, -0.262015, -0.063590, 0.231997, -0.602744, 0.116028, 0.010989, -0.070336, 0.022759, -0.008668, -0.293174, -0.097732, -0.045559, 0.058442, 0.113890, -0.037367, 0.094333, -0.069933, -0.006225, 0.031085, -0.106786, 0.119145, -0.392559, 0.237050, -0.390930, -0.118449, 0.214884, 0.027664, 0.169642, -0.049138, -0.197078, 0.054554, 0.061405, -0.357010, 0.107428, -0.283664, -0.549305, 0.196460, -0.066499, 0.098460, -0.127114, -0.030793, -0.331643, -0.003445, 0.031629, -0.054765, -0.143883, -0.078776, -0.505504, 0.205413, -0.151712, -0.309058, -0.072333, -0.182603, 0.081938, 0.184323, 0.057669, 0.086199, -0.093475, -0.014119, -0.109961, -0.086515, 0.057695, 0.140350, -0.073543, 0.038616, 0.093248, -0.257239, -0.097756, 0.238262, -0.192745, -0.019338, -0.427187, -0.168611, 0.286373, 0.005758, -0.113581, -0.002363, -0.443439, 0.348566, -0.437458, 0.063732, -0.004667, 0.218996, -0.452532, -0.008758, -0.255806, 0.154420, -0.193301, -0.043314, -0.000954, -0.028389, 0.119073, 0.008186, -0.015953, 0.089209, 0.096622, 0.109043, 0.118994, 0.049170, -0.127074, 0.105419, -0.384822, -0.327673, 0.026123, 0.017387, 0.130398, 0.100715, 0.047261, -0.043539, -0.280215, -0.040743, 0.127600, -0.468483, 0.102149, -0.257289, -0.628449, -0.081750, -0.111314, 0.196312, -0.115870, -0.139832, -0.416027, 0.096995, -0.243416, 0.249793, -0.099605, 0.163138, -0.349807, 0.073480, 0.422805, -0.607789, -0.183840, 0.076808, -0.379379, 0.225717, -0.205563, 0.035160, -0.080819, -0.066715, 0.106489, 0.030461, -0.145309, 0.053267, 0.074364, 0.057972, -0.032459, -0.144284, 0.249958, -0.145570, -0.335109, 0.307663, -0.515342, -0.156491, 0.147379, -0.016611, -0.378034, 0.293868, -0.348007, 0.503633, -0.366802, 0.232472, -0.106531, -0.077716, -0.546530, -0.033077, -0.146109, -0.053277, -0.092935, -0.054190, -0.045042, 0.005167, -0.077788, 0.120379, -0.045343, 0.122853, -0.144270, 0.185955, 0.065775, -0.036601, -0.414955, 0.083428, -0.406846, -0.064540, 0.127923, 0.103027, 0.164986, -0.015042, 0.032485, -0.224574, -0.168588, 0.029706, 0.104036, -0.301542, 0.160735, -0.263792, -0.312858, -0.155332, 0.114979, 0.235004, -0.350671, 0.121017, -0.254411, 0.062270, -0.023726, 0.073842, -0.601826, 0.827350, -0.930605, -0.524179, 0.350365, -0.531208, -0.143701, 0.538027, -0.258096, -0.069819, -0.110155, -0.120333, -0.121455, 0.087320, -0.034419, 0.036520, -0.257570, 0.080898, 0.093013, -0.046610, 0.078660, 0.052786, 0.070945, 0.007913, -0.218652, 0.024194, -0.040436, -0.047893, 0.118915, -0.051224, -0.234075, 0.378179, -0.118951, 0.263378, -0.633447, 0.253924, -0.042936, 0.158013, -1.192531, 0.050775, -0.138559, -0.117230, -0.183433, -0.148149, 0.063363, -0.165286, -0.169716, 0.153156, -0.075212, 0.167307, -0.244663, -0.283609, 0.423442, -0.051810, -0.496011, -0.127443, -0.171564, -0.229845, 0.333842, 0.242775, 0.159876, -0.101148, -0.035650, -0.120683, -0.094202, 0.049646, 0.061188, -0.118261, 0.025738, -0.199746, 0.060397, -0.381450, 0.073261, 0.178340, -0.013052, 0.348790, -0.035211, 0.003852, 0.070037, -0.259870, -0.752209, 0.087631, -0.886020, -0.952334, 0.493902, -0.277611, -0.389633, 0.504482, -0.463970, 0.339841, -0.104012, -0.021878, -0.282156, 0.147695, 0.092250, 0.036493, -0.202618, 0.019019, 0.135191, 0.001300, -0.029397, -0.275995, 0.377431, 0.025351, -0.298057, 0.020296, 0.103375, -0.072087, 0.284553, -0.215243, -0.065661, 0.333713, 0.003395, -0.018595, -0.587384, 0.389230, 0.023726, 0.122853, -1.121287, -0.058528, -0.699432, -0.105804, -0.438704, -0.271584, 0.018846, -0.317070, -0.305358, 0.293471, 0.003708, 0.383815, -0.035677, -0.622634, 0.034441, 0.371945, -0.320113, 0.032640, 0.166673, -0.229151, 0.116908, -0.237115, 0.080002, 0.052572, -0.091843, -0.164542, -0.054720, -0.053588, -0.066796, -0.139214, -0.163549, -0.031820, -0.214064, -0.369401, -0.234024, 0.359534, -0.345699, 0.196785, 0.151471, 0.048145, 0.201197, -0.021324, -1.185830, 0.232715, -0.857389, -0.511750, 0.094716, -0.368076, -0.883889, 0.318990, -0.467667, 0.556423, -0.073808, -0.238520, -0.143480, -0.095755, -0.022900, 0.247035, 0.021258, 0.071456, 0.071333, -0.131827, 0.067308, -0.255187, 0.397294, 0.083212, -0.098010, 0.002082, 0.183911, -0.003136, -0.195375, -0.069909, 0.354603, 0.309211, -0.169340, 0.155602, -0.374698, 0.380585, 0.044219, 0.027541, -0.856378, -0.102795, -0.974314, -0.028959, -0.425420, 0.000065, 0.108608, -0.013854, -0.358679, 0.220242, 0.120912, 0.057130, 0.177615, -0.519977, -0.079763, 0.330478, -0.577092, 0.064034, 0.098062, -0.280209, 0.170540, 0.106241, 0.077547, 0.191725, 0.183054, -0.189206, 0.009263, -0.281846, -0.068731, -0.287676, -0.157793, 0.048118, -0.260555, 0.021331, -0.067484, 0.256946, -0.086277, 0.413590, -0.325873, 0.009403, 0.102238, -0.364014, -1.052401, 0.016295, -0.756568, -0.290817, 0.179122, -0.400782, -0.140935, 0.219402, -0.120617, 0.284734, -0.119852, -0.100647, -0.124243, 0.217016, -0.028780, 0.112428, 0.077722, -0.195224, 0.056998, -0.198049, -0.129682, -0.187165, 0.117158, 0.091207, -0.269540, 0.033055, 0.041840, 0.029569, -0.114843, 0.254870, 0.262046, 0.390041, -0.233270, -0.126516, -0.007045, 0.381770, 0.104622, 0.108081, -0.394761, -0.232099, -0.791475, 0.456322, -0.101017, -0.015776, -0.072119, -0.197005, -0.340014, 0.257669, -0.286400, 0.127257, 0.095730, -0.101602, -0.286314, 0.063396, -0.179583, 0.220638, 0.145108, -0.359362, 0.346267, 0.117523, -0.155225, 0.032218, 0.146254, -0.287179, 0.176535, -0.325080, -0.012828, -0.170751, -0.014276, 0.048929, 0.126247, -0.026950, -0.051426, 0.504095, -0.018820, 0.146539, -0.052997, -0.078270, -0.404772, -0.013281, -0.512252, -0.398389, 0.003793, -0.006843, -0.126501, -0.343093, -0.038052, 0.007046, -0.037075, 0.320275, -0.218664, 0.072403, -0.191533, 0.132405, -0.064678, 0.083135, 0.159316, -0.064575, 0.162638, 0.054718, -0.346856, 0.035397, -0.091843, 0.096592, -0.157922, -0.062026, -0.093721, 0.150815, 0.142584, -0.078560, 0.218060, 0.352646, -0.150099, -0.429536, 0.044222, 0.062405, -0.074601, 0.212099, -0.106063, -0.143317, -0.586820, -0.227466, 0.126762, -0.013708, 0.311400, 0.055735, -0.752272, 0.165497, -0.485878, -0.037237, 0.013170, -0.113730, -0.314782, 0.261136, -0.115843, -0.076511, -0.244393, -0.118186, 0.318206, -0.062413, -0.028675, -0.046336, 0.067193, -0.176415, 0.278860, -0.234268, -0.135119, 0.020340, 0.025042, -0.134760, -0.142188, -0.009961, 0.140533, 0.050820, -0.044041, 0.228149, -0.184530, 0.111787, -0.177261, -0.385077, -0.378877, -0.084317, -0.254026, 0.069351, -0.300689, 0.021550, 0.073965, 0.091413, 0.072561, 0.232310, -0.310715, -0.264978, -0.088425, 0.091809, -0.044459, 0.002133, -0.377684, -0.132937, 0.121145, 0.167655, -0.283529, 0.061410, -0.147847, -0.022683, -0.105094, 0.027912, -0.052164, -0.267344, 0.088061, 0.340459, 0.296940, 0.245508, -0.070611, -0.240118, -0.040352, -0.081712, -0.020309, 0.172203, -0.140040, -0.063584, -0.489424, -0.482483, -0.079252, 0.006029, 0.106248, 0.327293, -0.601155, -0.422981, -0.353093, 0.257653, -0.257056, -0.013784, -0.169271, 0.098763, 0.000354, -0.004363, -0.292679, -0.324301, 0.071339, 0.227930, -0.091788, 0.027340, 0.070473, -0.233851, 0.145878, -0.159731, -0.263973, 0.073026, -0.139778, 0.277370, -0.219214, 0.114166, 0.002204, -0.250626, 0.073715, 0.247415, -0.117056, -0.299691, -0.252960, 0.139221, -0.710512, 0.214996, -0.118943, -0.435635, -0.287442, -0.071865, 0.086349, 0.138027, -0.093329, 0.334603, -0.391011, -0.161164, -0.129170, -0.074026, 0.091924, 0.215390, -0.443662, -0.074110, -0.299017, 0.346636, -0.131584, -0.264273, -0.238851, 0.142892, 0.100447, 0.023454, -0.169982, -0.109035, -0.055376, 0.029334, 0.180339, 0.379788, -0.090010, -0.319764, 0.014851, -0.086614, -0.031928, 0.098296, -0.093678, -0.142736, -0.132314, 0.139826, -0.213443, -0.210325, -0.190687, 0.249232, -0.236017, 0.051969, -0.375890, -0.127350, -0.059540, -0.097091, -0.014688, -0.078542, -0.126943, 0.153290, -0.116557, -0.188076, -0.334756, -0.035096, -0.097091, 0.228246, 0.067502, -0.356533, -0.082132, 0.035794, -0.349333, 0.174077, -0.360900, 0.331900, -0.151440, -0.250649, 0.227000, -0.254082, 0.153085, 0.056047, -0.295726, -0.257117, -0.206484, -0.115717, -0.312759, -0.334970, -0.132648, 0.005161, 0.058808, 0.164607, -0.243542, -0.122173, 0.002794, -0.090739 
bias_vector:
.float 0.187204, 0.311646, 0.024405, -0.276411, -0.002305, 0.018354, 0.012592, 0.113989, -0.153160, -0.077566
   

final_output:
    .rept 10              # final output vector
        .float 0.0
    .endr

intermediate:
    .rept 10              
        .float 0.0
    .endr


probability_matrix:
    .rept 10
        .float 0.0
    .endr
  
