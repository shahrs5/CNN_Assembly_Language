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


    la   a0, flattened_pool # Load address of pooled feature maps
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


#old and incorrect 
#----------------------------------------------------------------------
# denselayer: Fully connected layer (dense layer) implementation
#             Performs: output[i] = dot(input, weights[i][:]) + bias[i]
#
# Inputs:
#   a0 = base pointer to input vector (flattened 12x12 = 144 float32)
#   a1 = base pointer to weight matrix (10x1152, row-major)
#   a2 = base pointer to bias vector (10 float32 values)
#   a3 = base pointer to output vector (10 float32 outputs)
#
# Weight matrix layout: 10 rows × 1152 columns
# Each row contains 1152 weights for one output neuron
# Special structure: 8 blocks of 144 elements (12×12) per row
#
# Registers:
#   s0 = pointer to input vector (reset per neuron)
#   s1 = pointer to weight matrix base
#   s2 = pointer to bias vector
#   s3 = pointer to output vector
#   s4 = working pointer for current weight row
#   t0 = outer loop counter (for neurons 0 to 9)
#   t1 = inner block counter (0 to 7, for 8 blocks of 144)
#   t2 = constant 10 (number of output neurons)
#   t3 = temporary calculation register
#   t4 = elements remaining in current block
#   t5 = vector length for current iteration
#   v0 = vector register for input vector
#   v1 = vector register for weights
#   v2 = vector product of v0 and v1
#   v3 = accumulator for dot product

# .globl denselayer       # Make denselayer function globally accessible
# denselayer:             # Start of dense layer function
#     mv s0, a0           # input vector pointer
#     mv s1, a1           # weight matrix base pointer
#     mv s2, a2           # bias vector pointer
#     mv s3, a3           # output vector pointer
#     li t2, 10           # number of output neurons
#     li t0, 0            # initialize outer loop counter

# dense_outer:
#     # Initialize accumulator for current neuron
#     vsetvli t3, zero, e32
#     vmv.v.i v3, 0       # clear accumulator
    
#     # Calculate pointer to current weight row: weights[t0][0]
#     li t3, 1152         # elements per row
#     slli t3, t3, 2      # convert to bytes (1152 * 4)
#     mul t3, t0, t3      # row offset in bytes
#     add s4, s1, t3      # s4 = &weights[t0][0]
    
#     # Reset input pointer for this neuron
#     mv s0, a0
    
#     # Process 8 blocks of 144 elements each (8 * 144 = 1152)
#     li t1, 0            # block counter
    
# dense_block_loop:
#     # Process one block of 144 elements
#     li t4, 144          # elements in current block
    
# dense_inner:
#     vsetvli t5, t4, e32 # set vector length to min(vlen, remaining elements)
#     sub t4, t4, t5      # update remaining elements in block
#     slli t3, t5, 2      # calculate byte offset for this vector
    
#     # Load input vector elements
#     vle32.v v0, (s0)
#     add s0, s0, t3      # advance input pointer
    
#     # Load corresponding weight elements
#     vle32.v v1, (s4)
#     add s4, s4, t3      # advance weight pointer
    
#     # Multiply and accumulate
#     vfmul.vv v2, v1, v0     # element-wise multiply
#     vfredosum.vs v3, v2, v3 # reduce sum into accumulator
    
#     bnez t4, dense_inner    # continue if elements remain in block
    
#     # Move to next block
#     addi t1, t1, 1
#     li t3, 8
#     blt t1, t3, dense_block_loop  # process next block if t1 < 8
    
#     # Add bias and store result
#     slli t3, t0, 2      # calculate bias offset
#     add t3, s2, t3      # pointer to bias[t0]
#     flw f0, (t3)        # load bias value
#     vfmv.f.s f1, v3     # move accumulator to scalar register
#     fadd.s f1, f1, f0   # add bias
    
#     # Store result
#     slli t3, t0, 2      # calculate output offset
#     add t3, s3, t3      # pointer to output[t0]
#     fsw f1, (t3)        # store result
    
#     # Next neuron
#     addi t0, t0, 1
#     blt t0, t2, dense_outer  # continue if t0 < 10

# done_outer:
#     ret 

#new and improved -- to be tested
#----------------------------------------------------------------------
# denselayer: fully connected layer implementation
#             Performs: output[i] = dot(input, weights[i]) + bias[i]
#
# Inputs:
#   a0 = base pointer to input vector (flattened, size determined by application)
#   a1 = base pointer to weight matrix (10×1152, row-major format)
#   a2 = base pointer to bias vector (10 float32 values)
#   a3 = base pointer to output vector (10 float32 outputs)
#   a4 = input size (1152 for this implementation)
#
# Weight matrix layout (row-major):
#   weights[0][0], weights[0][1], ..., weights[0][1151],
#   weights[1][0], weights[1][1], ..., weights[1][1151],
#   ...
#   weights[9][0], weights[9][1], ..., weights[9][1151]
#
# Registers:
#   s0 = pointer to input vector (reset per neuron)
#   s1 = pointer to current weight row
#   s2 = pointer to bias vector
#   s3 = pointer to output vector
#   s4 = input size (1152)
#   t0 = outer loop counter (neurons 0-9)
#   t1 = inner loop remaining elements
#   t2 = vector length for current iteration
#   t3 = temporary for address calculations
#   v0 = input vector chunk
#   v1 = weight vector chunk
#   v2 = element-wise products
#   v3 = dot product accumulator

.globl denselayer
denselayer:
    # Save callee-saved registers
    addi sp, sp, -32
    sw s0, 0(sp)
    sw s1, 4(sp)
    sw s2, 8(sp)
    sw s3, 12(sp)
    sw s4, 16(sp)
    sw ra, 20(sp)
    
    # Initialize pointers and constants
    mv s0, a0                # input vector base
    mv s1, a1                # weight matrix base
    mv s2, a2                # bias vector base
    mv s3, a3                # output vector base
    mv s4, a4                # input size (1152)
    
    li t0, 0                 # outer loop counter (neuron index)
    
neuron_loop:
    # Initialize accumulator for this neuron
    vsetvli t2, zero, e32, m1
    vmv.v.i v3, 0            # clear accumulator
    
    # Reset pointers for this neuron
    mv a0, s0                # reset input pointer
    mv a1, s1                # current weight row pointer
    mv t1, s4                # remaining elements to process
    
inner_loop:
    vsetvli t2, t1, e32, m1    # Set vector length for this iteration
    vle32.v v0, (a0)           # Load input chunk
    vle32.v v1, (a1)           # Load corresponding weight chunk (sequential load)
    vfmul.vv v2, v0, v1        # Compute element-wise products
    vfredosum.vs v3, v2, v3    # Accumulate into dot product
    
    # Update pointers and counter
    slli t3, t2, 2           # t3 = vector_length * 4 (bytes)
    add a0, a0, t3           # advance input pointer
    add a1, a1, t3           # advance weight pointer
    sub t1, t1, t2           # decrease remaining elements
    
    # Continue if more elements to process
    bnez t1, inner_loop
    
    # Extract scalar result from accumulator
    vfmv.f.s f0, v3
    
    # Add bias
    flw f1, (s2)
    fadd.s f0, f0, f1
    
    # Store result
    fsw f0, (s3)
    
    # Move to next neuron
    addi t0, t0, 1           # increment neuron counter
    addi s2, s2, 4           # next bias element
    addi s3, s3, 4           # next output element
    
    # Calculate next weight row address
    # Next row = current_row + input_size * 4 bytes
    slli t3, s4, 2           # input_size * 4
    add s1, s1, t3           # move to next weight row
    
    # Check if more neurons to process
    li t3, 10
    blt t0, t3, neuron_loop
    
    # Restore callee-saved registers
    lw s0, 0(sp)
    lw s1, 4(sp)
    lw s2, 8(sp)
    lw s3, 12(sp)
    lw s4, 16(sp)
    lw ra, 20(sp)
    addi sp, sp, 32
    
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
.float 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000
.float 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000
.float 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000
.float 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000
.float 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000
.float 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.051, 0.098, 0.392, 0.478, 0.027, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000
.float 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.129, 0.592, 0.816, 0.988, 0.988, 0.988, 0.573, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000
.float 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.157, 0.596, 0.957, 0.988, 0.992, 0.878, 0.827, 0.988, 0.910, 0.157, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000
.float 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.059, 0.596, 0.937, 0.988, 0.988, 0.988, 0.847, 0.122, 0.145, 0.988, 0.988, 0.235, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000
.float 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.376, 0.988, 0.988, 0.988, 0.988, 0.851, 0.114, 0.000, 0.145, 0.988, 0.988, 0.235, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000
.float 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.710, 0.988, 0.988, 0.863, 0.655, 0.118, 0.000, 0.000, 0.302, 0.988, 0.988, 0.235, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000
.float 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.102, 0.502, 0.227, 0.086, 0.000, 0.000, 0.000, 0.000, 0.392, 0.988, 0.988, 0.235, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000
.float 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.616, 0.988, 0.988, 0.235, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000
.float 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.431, 0.475, 0.478, 0.475, 0.792, 0.988, 0.761, 0.012, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000
.float 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.039, 0.208, 0.702, 0.992, 0.992, 1.000, 0.992, 0.992, 0.894, 0.137, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000
.float 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.020, 0.212, 0.890, 0.988, 0.953, 0.894, 0.667, 0.949, 0.988, 0.988, 0.906, 0.459, 0.024, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000
.float 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.024, 0.306, 0.988, 0.988, 0.490, 0.231, 0.000, 0.071, 0.816, 0.988, 0.988, 0.988, 0.988, 0.341, 0.027, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000
.float 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.020, 0.529, 0.988, 0.988, 0.706, 0.063, 0.000, 0.082, 0.796, 0.992, 0.969, 0.506, 0.678, 0.988, 0.988, 0.722, 0.259, 0.192, 0.192, 0.000, 0.000, 0.000
.float 0.000, 0.000, 0.000, 0.000, 0.000, 0.012, 0.533, 0.988, 0.945, 0.416, 0.067, 0.000, 0.208, 0.784, 0.988, 0.847, 0.255, 0.000, 0.055, 0.282, 0.639, 0.945, 0.988, 0.988, 0.875, 0.000, 0.000, 0.000
.float 0.000, 0.000, 0.000, 0.000, 0.000, 0.412, 0.988, 0.949, 0.345, 0.071, 0.286, 0.667, 0.957, 0.988, 0.494, 0.114, 0.000, 0.000, 0.000, 0.000, 0.000, 0.349, 0.706, 0.706, 0.145, 0.000, 0.000, 0.000
.float 0.000, 0.000, 0.000, 0.000, 0.000, 0.906, 0.988, 0.961, 0.804, 0.847, 0.988, 0.988, 0.988, 0.486, 0.012, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000
.float 0.000, 0.000, 0.000, 0.000, 0.000, 0.812, 0.988, 0.988, 0.988, 0.988, 0.698, 0.455, 0.141, 0.016, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000
.float 0.000, 0.000, 0.000, 0.000, 0.000, 0.051, 0.365, 0.561, 0.475, 0.090, 0.024, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000
.float 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000
.float 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000
.float 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000
.float 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000
.float 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000
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
.float -0.208,-0.123,-0.051,-0.373,0.232,0.094,0.016,-0.048,-0.018,-0.220,-0.255,-0.124,0.233,0.095,-0.109,-0.239,-0.177,-0.145,-0.084,0.160,0.078,-0.040,-0.135,-0.037,-0.347,0.047,-0.202,0.097,-0.068,0.074,-0.460,-0.205,-0.165,-0.006,0.163,0.035,-0.148,-0.019,0.024,-0.191,0.036,-0.066,0.187,0.126,-0.189,-0.002,-0.015,-0.071,0.016,-0.007,0.106,0.085,-0.111,-0.113,-0.047,-0.153,0.019,0.150,-0.191,0.131,-0.098,-0.011,-0.015,0.040,-0.116,0.078,-0.012,0.215,0.003,-0.034,-0.137,0.072,-0.162,-0.043,0.098,0.059,-0.158,0.013,-0.246,0.069,-0.161,0.142,-0.009,0.102,0.121,-0.065,-0.211,0.153,0.244,-0.209,0.021,-0.123,0.093,0.092,-0.025,-0.184,-0.086,-0.065,-0.005,0.009,-0.217,0.106,-0.123,0.037,-0.352,-0.067,-0.338,0.098,-0.555,-0.045,-0.221,-0.464,0.109,-0.257,-0.220,-0.034,-0.547,-0.052,-0.050,-0.059,0.051,-0.185,-0.484,-0.019,-0.461,-0.078,0.117,0.157,-0.148,-0.189,0.030,-0.092,-0.462,-0.117,-0.055,-0.031,-0.135,-0.041,-0.117,-0.108,-0.248,-0.211,-0.081,-0.078,0.027,-0.109,0.063,0.103,-0.153,-0.131,0.073,-0.045,0.018,-0.163,-0.026,0.088,0.038,-0.500,-0.032,0.090,-0.179,0.140,0.016,0.098,-0.092,-0.472,-0.021,-0.050,0.226,0.032,-0.000,0.112,-0.217,-0.361,0.032,-0.129,-0.006,0.385,0.175,0.037,-0.345,-0.295,0.143,0.271,-0.012,-0.122,0.049,0.204,-0.136,0.086,0.042,-0.171,0.131,-0.016,-0.104,-0.006,0.139,-0.020,-0.046,-0.188,-0.002,-0.349,-0.426,0.057,-0.250,-0.029,0.107,0.071,-0.050,0.020,-0.330,-0.017,-0.308,-0.089,0.003,0.007,-0.102,-0.097,-0.190,-0.095,-0.480,-0.323,-0.083,-0.072,-0.269,-0.143,-0.034,-0.065,-0.515,-0.332,0.116,-0.036,0.085,-0.071,-0.032,0.170,-0.173,-0.235,-0.005,0.055,-0.179,0.007,-0.001,0.189,0.028,-0.068,-0.052,-0.053,-0.125,0.043,-0.027,0.127,0.192,-0.471,-0.109,0.153,0.096,0.285,0.016,-0.003,-0.015,-0.595,-0.206,-0.066,-0.039,0.107,0.038,0.069,-0.112,-0.523,0.103,-0.052,-0.177,0.266,0.099,-0.010,-0.359,-0.431,-0.082,-0.168,-0.068,0.103,0.095,0.172,-0.016,0.032,0.166,-0.420,-0.004,0.031,0.012,-0.022,-0.016,0.032,0.003,-0.068,0.082,-0.374,-0.283,0.140,-0.158,0.051,0.096,-0.079,0.001,-0.009,-0.054,-0.075,-0.197,-0.062,0.017,-0.085,-0.203,0.039,-0.082,-0.027,-0.278,-0.079,-0.049,-0.132,-0.077,-0.153,0.002,-0.006,-0.091,0.304,-0.073,0.017,0.006,-0.176,-0.012,0.145,-0.084,-0.079,-0.054,-0.050,0.093,0.113,0.080,0.100,-0.111,-0.261,-0.158,-0.146,0.056,0.086,0.149,0.039,-0.004,-0.205,-0.094,-0.055,0.136,0.168,0.116,0.118,0.048,-0.519,-0.127,-0.039,-0.237,0.173,0.068,0.099,-0.143,-0.523,-0.139,-0.016,-0.184,0.063,0.042,0.077,-0.202,-0.229,-0.094,0.108,-0.161,0.110,0.074,0.089,-0.116,0.032,-0.162,-0.169,0.025,-0.001,-0.098,-0.004,0.097,0.025,-0.039,0.050,0.086,-0.234,0.016,0.070,-0.132,0.052,-0.162,-0.052,0.078,-0.025,-0.039,0.098,-0.347,0.053,0.160,0.023,-0.041,-0.149,0.208,-0.006,-0.054,0.195,0.027,-0.070,-0.041,0.003,0.033,-0.034,-0.125,0.113,-0.040,-0.176,0.031,-0.057,0.001,-0.144,-0.160,-0.021,-0.052,-0.023,0.196,0.219,0.260,-0.367,-0.007,-0.058,-0.063,0.127,-0.207,0.225,0.074,-0.317,-0.062,-0.151,-0.248,0.020,-0.019,0.005,0.276,-0.180,-0.098,-0.406,-0.411,0.230,-0.298,0.109,0.118,-0.012,-0.149,-0.337,-0.587,0.152,-0.052,0.111,0.073,0.090,-0.482,-0.143,-0.217,0.348,0.169,0.068,0.165,-0.016,-0.272,0.042,-0.326,0.502,-0.101,-0.033,0.120,0.026,-0.118,0.122,-0.285,0.169,0.096,-0.296,-0.081,-0.222,-0.166,0.086,0.031,0.045,0.270,0.118,0.039,-0.135,-0.469,0.143,0.179,0.113,0.117,0.074,0.105,0.131,-0.285,0.105,0.007,0.013,0.208,0.112,0.189,0.025,-0.241,-0.060,0.074,0.218,-0.119,0.108,0.210,-0.201,-0.368,-0.036,-0.136,-0.019,0.003,0.145,0.278,-0.695,-0.041,0.069,-0.166,0.042,-0.046,0.009,0.149,-0.422,0.025,-0.006,-0.570,0.100,-0.283,-0.010,0.052,-0.287,-0.062,-0.076,-0.377,0.020,-0.218,0.065,0.106,-0.027,0.085,-0.029,-0.176,0.269,0.039,0.110,0.116,0.066,-0.363,-0.095,-0.257,0.231,-0.035,0.011,0.086,0.063,-0.390,-0.024,-0.313,0.258,-0.057,0.003,-0.162,-0.036,0.003,0.019,-0.318,0.064,0.141,-0.112,0.025,-0.131,-0.053,-0.046,0.180,0.168,0.074,0.127,0.252,-0.184,-0.357,0.110,0.101,0.236,0.058,0.011,0.262,-0.236,-0.351,0.105,0.133,0.291,0.057,0.146,0.227,-0.205,-0.164,0.156,-0.026,0.225,-0.307,0.101,0.375,-0.128,-0.367,0.206,-0.072,0.008,0.038,0.102,0.084,-0.242,-0.383,0.036,-0.450,-0.238,0.047,-0.312,-0.059,-0.323,-0.169,-0.018,-0.342,0.126,0.099,-0.142,0.154,-0.161,0.052,0.048,-0.242,0.117,0.012,0.020,0.022,-0.103,0.088,0.138,-0.012,0.085,0.302,0.162,0.033,-0.187,-0.079,0.059,0.030,0.180,-0.052,0.171,0.063,0.064,-0.129,0.089,-0.100,0.199,0.022,-0.419,-0.160,0.067,0.026,-0.028,-0.161,-0.140,0.241,-0.008,0.027,-0.214,-0.013,0.065,0.094,0.160,0.092,0.138,0.125,-0.319,-0.102,-0.074,0.028,0.184,0.106,0.041,0.156,-0.112,-0.207,-0.002,-0.063,0.232,-0.186,0.095,0.311,-0.183,-0.496,0.073,0.086,0.126,0.117,0.035,0.403,0.024,-0.547,0.147,-0.186,0.078,-0.121,-0.151,-0.049,-0.047,-0.531,-0.404,-0.164,-0.178,0.164,-0.332,-0.143,-0.177,-0.248,0.165,-0.163,0.004,0.080,-0.146,-0.076,-0.145,-0.055,0.242,-0.039,0.038,-0.016,0.136,0.069,-0.069,0.007,0.119,-0.082,-0.075,-0.070,0.169,0.080,-0.266,-0.054,0.147,0.006,-0.025,-0.240,0.180,0.045,-0.123,0.007,0.085,-0.150,0.095,0.070,-0.136,0.146,0.046,0.036,-0.104,0.326,-0.250,0.135,0.015,0.422,-0.265,-0.141,-0.093,0.007,0.018,0.095,0.028,-0.006,-0.149,0.045,-0.446,-0.082,0.102,0.108,0.081,0.206,-0.100,0.077,-0.212,-0.083,0.145,-0.158,0.106,0.253,0.071,0.007,-0.017,-0.018,0.114,-0.021,0.115,0.229,0.197,-0.119,-0.077,0.022,0.105,0.100,-0.419,-0.199,0.135,0.007,0.136,0.008,0.165,0.004,-0.320,-0.115,0.074,0.100,0.184,-0.096,-0.005,0.058,-0.108,-0.115,-0.054,0.117,0.058,0.090,0.150,0.113,-0.151,-0.066,-0.180,0.020,0.092,0.158,0.189,-0.223,-0.073,0.057,-0.412,0.116,0.142,0.005,0.027,-0.610,-0.074,0.008,-0.077,0.013,0.107,-0.297,-0.118,0.083,0.120,0.199,-0.459,0.071,0.073,0.164,-0.085,-0.044,0.198,0.224,-0.468,0.015,-0.249,-0.129,0.062,-0.307,0.083,0.114,-0.404,-0.043,-0.202,-0.093,0.109,0.036,0.134,0.321,-0.066,0.012,-0.206,-0.070,0.039,-0.077,0.173,0.288,0.095,0.025,-0.273,-0.032,0.146,0.115,0.182,0.075,0.235,0.096,0.056,0.111,0.227,0.094,-0.127,-0.347,0.359,-0.048,0.096,-0.095,0.134,-0.067,-0.459,-0.340,0.069,-0.122,0.088,0.028,-0.057,-0.022,-0.272,-0.146,-0.042,-0.091,0.032,0.024,-0.098,-0.162,-0.298,-0.206,-0.019,0.020,0.013,0.022,-0.130,-0.215,-0.290,-0.050,-0.168,-0.031,0.104,0.110,-0.078,-0.134,-0.232,-0.061,-0.290,-0.049,0.054,0.056,-0.276,0.044,-0.088,0.039,-0.144,-0.027,0.019,-0.048,0.036,-0.080,0.302,0.189,-0.583,-0.035,-0.008,0.098,-0.007,-0.098,0.251,0.285,-0.210,-0.024,-0.149,0.046,-0.027,0.015,0.317,0.239,-0.148,0.030,-0.120,-0.103,-0.250,0.314,0.421,0.052,0.263,0.102,-0.058,0.131,-0.148,0.089,0.289,-0.251,0.368,0.209,-0.259,0.025,-0.349,0.100,-0.186,-0.428,0.246,0.056,-0.191,0.140,-0.342,-0.060,-0.361,-0.310,0.087,-0.005,-0.089,-0.001,-0.363,-0.046,-0.365,-0.249,-0.257,-0.042,0.004,-0.042,-0.461,0.238,-0.385,-0.422,-0.518,-0.079,-0.013,0.234,-0.197,-0.066,0.076,-0.066,-0.640,-0.108,-0.089,0.181,-0.264,0.093,-0.268,-0.102,-0.349,-0.120,0.011,0.139,-0.423,-0.058,-0.030,0.054,-0.459,-0.031,-0.009,0.015,0.125,0.194,0.030,0.141,-0.671,0.143,-0.082,0.205,0.177,0.143,0.017,-0.009,-0.952,0.159,0.039,-0.297,0.076,-0.066,-0.299,-0.303,-0.673,0.062,0.037,-0.316,-0.038,0.093,-0.260,-0.313,-0.082,0.140,0.021,-0.210,-0.282,-0.125,-0.265,-0.384,-0.011,0.114,-0.266,-0.161,-0.210,-0.652,-0.392,-0.557,-0.267,0.103,-0.123,0.141,-0.464,-0.436,-0.208,-0.411,-1.068,0.013,-0.037,0.135,-0.654,-0.501,-0.153,-0.229,-0.609,-0.074,0.025,-0.179,-0.565,-0.429,-0.440,-0.478,-0.545,-0.011,-0.150,-0.072,-0.461,-0.574,-0.110,-0.286,-0.699,-0.064,-0.162,-0.322,-0.509,-0.203,-0.304,-0.300,-0.298,-0.080,-0.004,-0.129,-0.376,0.320,0.046,0.265,0.303,0.295,0.183,-0.003,0.096,-0.017,0.117,0.173,0.017,0.256,0.158,-0.055,-0.042,-0.029,-0.160,0.258,-0.075,-0.090,0.149,-0.189,-0.233,0.208,0.197,0.264,0.024,0.100,0.142,-0.421,0.032,0.135,0.077,0.202,-0.066,-0.042,0.123,-0.467,-0.103,-0.151,0.188,-0.105,-0.111,-0.460,0.291,-0.330,0.287,-0.051,0.066,-0.039,-0.098,-0.714,0.284,0.154,0.095,-0.150,-0.188,0.056,-0.067,-0.482,0.315,-0.120,0.041,-0.097,-0.068,-0.046,-0.063,0.306,0.187,0.167,0.258,0.146,0.017,0.060,-0.008,0.197,0.298,0.089,0.152,-0.027,0.062,-0.009,-0.043,0.242,0.156,-0.039,-0.103,-0.256,-0.048,-0.122,-0.092,0.271,0.144,-0.224,-0.097,0.185,-0.227,0.076,0.117,0.376,0.102,-0.056,0.073,-0.094,0.103,-0.087,0.026,-0.024,0.131,-0.182,0.016,0.033,0.352,0.357,-0.157,0.287,0.297,-0.347,0.144,-0.097,0.101,0.379,0.002,0.092,0.248,-0.294,-0.050,-0.303,0.446,0.348,-0.248,-0.350,0.369,-0.336,0.165,-0.072,-0.008,0.235,-0.147,-0.474,0.394,-0.166,0.260,-0.065,0.242,0.126,-0.218,-0.189,0.579,-0.060,0.150,0.039,0.115,0.115,-0.067,-0.222,0.208,-0.098,0.214,0.095,0.009,0.060,-0.009,-0.523,0.178,0.104,0.267,-0.128,-0.078,0.168,0.010,-0.182,0.179,0.107,0.147,-0.008,-0.130,0.167,0.028,-0.105,0.073,0.118,0.213,-0.048,0.002,0.014,-0.078,-0.381,0.121,0.024,-0.154,0.229,-0.207,-0.107,-0.040,0.153,0.186,0.007,-0.086,0.181,-0.130,0.082,0.070,0.048,0.125,-0.028,0.289,-0.087,0.346,0.383,0.041,-0.010,0.135,-0.308,-0.111,-0.037,0.062,0.195,-0.293,0.088,-0.044,-0.322,-0.489,-0.094,0.332,0.233,-0.251,-0.181,0.070,-0.197,0.111,-0.155,0.080,0.240,-0.251,-0.304,0.414,-0.404,0.082,-0.060,0.087,0.210,-0.280,-0.185,0.411,-0.223,0.126,-0.256,-0.079,0.044,-0.143,-0.441,0.220,-0.183,0.138,-0.056,-0.043,0.061,-0.022,-0.496,0.251,0.067,0.429,0.006,0.051,0.123,0.026,-0.412,0.239,0.223,0.321,0.097,0.191,0.261,0.026,-0.329,0.103,0.064,0.333,-0.292,-0.125,0.018,-0.012,-0.420,0.054,-0.184,-0.487,-0.162,-0.342,-0.225,-0.331,-0.151,0.204,-0.347,-0.292,0.059,0.043,0.186,0.086,0.166,0.141,-0.138,-0.061,-0.306,0.319,0.108,0.060,-0.039,-0.003,-0.262,-0.017,0.136,0.085,-0.184,-0.097,0.044,-0.252,0.005,-0.278,0.204,0.086,-0.050,-0.342,-0.144,0.012,-0.041,-0.166,0.070,-0.188,-0.109,-0.317,-0.095,0.066,-0.214,-0.296,-0.120,-0.027,0.091,-0.183,-0.058,0.235,-0.076,-0.167,-0.501,0.036,0.012,-0.145,-0.297,-0.088,-0.141,-0.057,-0.300,0.065,0.098,-0.138,-0.663,-0.031,-0.012,-0.024,-0.186,-0.133,-0.005,-0.036,-0.216,0.152,0.198,-0.071,-0.252,0.193,0.205,-0.014,-0.054,0.122,-0.117,0.137,-0.417,-0.138,0.089,0.093,-0.292,0.015,-0.222,-0.504,-0.340,-0.202,-0.235,-0.035,-0.188,0.142,-0.059,-0.208,0.200,0.126,0.275,-0.248,0.141,0.159,0.119,-0.188,0.116,-0.445,-0.142,-0.066,0.040,0.127,0.182,0.054,-0.196,-0.256,-0.696,-0.203,0.076,0.157,-0.117,-0.501,-0.003,-0.141,-0.786,-0.197,-0.124,0.152,0.067,-0.341,0.072,-0.286,-0.090,-0.100,-0.130,-0.156,-0.070,-0.337,0.051,-0.049,0.132,-0.266,-0.058,0.222,-0.194,-0.049,-0.084,-0.001,0.157,-0.090,-0.239,-0.012,-0.079,0.024,-0.606,0.136,0.087,-0.109,-0.436,0.059,-0.150,-0.184,-0.357,0.100,-0.038,0.073,-0.256,0.130,-0.030,-0.211,0.140,0.080,0.170,-0.133,-0.081,0.101,-0.031,0.008,-0.043,-0.053,0.120,-0.175,-0.008,0.078,-0.255,-0.347,-0.056,-0.232,-0.215,0.172,-0.045,0.207,0.056,0.223,-0.254,0.013,0.273,-0.297,-0.027,0.261,0.003,-0.060,-0.061,-0.330,-0.337,-0.107,-0.231,0.194,-0.205,-0.004,0.005,-0.772,-0.434,-0.161,-0.022,0.274,0.060,-0.269,0.174,-0.814,-0.475,-0.259,-0.234,0.584,0.058,-0.183,-0.003,-0.196,0.020,-0.137,0.037,0.427,0.017,-0.208,0.112,0.147,0.219,-0.266,-0.391,0.146,-0.229,0.016,-0.429,0.122,0.224,0.054,-0.205,0.037,-0.183,0.013,-0.483,-0.096,0.091,-0.238,-0.162,0.049,-0.101,-0.130,-0.191,0.065,0.010,-0.306,0.033,0.129,-0.191,0.290,-0.300,0.048,0.088,-0.174,0.026,0.208,-0.384,-0.156,-0.189,-0.209,-0.223,-0.155,0.089,0.094,-0.468,-0.135,0.228,-0.273,0.141,-0.196,0.146,0.223,0.023,0.283,0.010,-0.327,0.163,-0.266,-0.025,0.272,-0.016,0.150,-0.040,-0.000,-0.097,-0.259,0.053,0.323,-0.104,-0.376,0.075,-0.296,-0.394,-0.146,-0.132,0.281,0.014,-0.169,0.012,-0.556,-0.330,-0.462,-0.063,0.673,-0.190,-0.215,0.089,-0.029,-0.008,-0.562,-0.057,0.264,-0.116,0.191,-0.156,0.006,0.128,-0.406,-0.111,0.188,-0.267,0.239,-0.288,0.165,0.165,-0.039,-0.307,0.201,-0.126,0.208,-0.862,-0.088,0.127,-0.107,-0.225,0.156,-0.283,0.086,-0.176,-0.186,0.025,-0.112,0.035,-0.098,-0.262,0.209,-0.263,-0.286,-0.187,-0.128,0.115,0.176,-0.267,-0.178,-0.181,-0.219,-0.185,-0.009,0.177,-0.007,-0.118,-0.394,-0.103,-0.286,-0.060,-0.340,0.216,0.280,-0.393,-0.025,-0.081,-0.264,-0.462,-0.096,-0.112,0.381,-0.059,-0.264,-0.003,-0.110,-0.680,0.185,0.008,0.221,0.135,-0.412,-0.147,-0.137,-0.267,-0.177,-0.144,0.614,-0.332,-0.383,-0.138,-0.280,-0.196,-0.393,0.127,0.429,-0.123,-0.112,-0.111,-0.011,0.021,-0.615,0.123,0.709,-0.211,0.052,-0.502,0.198,0.149,-0.042,-0.011,0.365,-0.242,0.181,-0.300,0.246,0.154,-0.061,-0.193,0.191,-0.077,-0.174,-0.180,0.078,0.017,-0.083,-0.302,0.070,0.112,0.090,0.138,-0.042,0.046,0.005,-0.103,-0.199,0.049,-0.236,0.095,-0.019,-0.148,0.125,0.282,-0.009,-0.052,-0.417,0.007,0.077,-0.079,-0.017,0.173,-0.082,-0.164,-0.013,-0.113,0.070,0.079,-0.235,0.020,0.131,-0.462,-0.071,-0.115,-0.174,-0.068,0.191,-0.001,0.183,-0.009,-0.106,0.111,-0.061,-0.660,0.203,-0.025,0.402,0.175,-0.305,0.162,-0.091,-0.207,0.215,-0.017,0.494,0.038,-0.314,-0.260,-0.168,0.005,-0.000,-0.028,0.475,-0.175,-0.361,-0.056,-0.162,0.104,-0.193,0.168,0.684,-0.258,-0.123,-0.164,0.144,0.035,-0.068,-0.069,0.280,-0.191,0.101,0.014,0.127,0.073,0.055,0.011,-0.248,0.068,-0.232,0.019,-0.003,0.105,0.105,-0.228,-0.281,0.021,-0.074,-0.009,0.032,-0.107,0.151,0.161,-0.199,-0.085,-0.399,-0.010,0.036,-0.239,0.155,0.157,-0.191,-0.027,-0.325,0.143,0.036,0.092,0.096,0.052,0.055,0.094,0.019,0.070,-0.088,-0.004,0.072,-0.207,0.187,0.046,-0.248,0.249,0.048,-0.297,0.089,-0.140,0.087,0.127,-0.153,0.306,-0.213,-0.281,0.204,0.035,0.279,0.200,-0.129,-0.022,-0.153,-0.111,-0.002,-0.053,0.277,0.031,-0.375,-0.165,-0.387,-0.039,-0.097,-0.066,0.275,-0.087,-0.252,-0.093,-0.136,0.027,-0.158,-0.072,0.047,-0.013,-0.063,-0.161,0.028,-0.034,-0.217,0.045,-0.099,0.021,-0.049,-0.164,0.185,0.072,-0.030,0.044,-0.230,0.122,-0.462,-0.163,0.230,0.033,0.050,0.219,-0.139,0.039,-0.499,0.153,0.105,-0.077,0.106,0.197,-0.112,0.187,-0.592,0.153,-0.042,-0.088,-0.017,0.192,-0.003,0.184,-0.059,-0.251,-0.011,-0.001,-0.023,0.179,0.064,-0.106,0.083,-0.167,-0.089,-0.035,-0.421,-0.081,0.204,-0.083,0.000,-0.068,-0.292,-0.202,-0.316,0.032,0.358,0.047,0.011,0.089,0.093,-0.024,-0.092,-0.084,0.231,-0.037,-0.095,-0.242,-0.101,0.025,-0.065,-0.039,0.160,-0.029,0.017,-0.136,-0.039,0.157,-0.304,0.020,0.123,-0.081,-0.090,-0.222,0.123,0.112,-0.482,0.062,-0.110,-0.154,-0.043,-0.374,0.155,0.031,-0.034,0.152,0.091,-0.179,-0.146,-0.539,0.013,0.101,-0.079,0.232,0.109,-0.246,-0.423,-0.346,0.147,0.044,0.114,0.206,0.001,-0.313,0.046,-0.373,0.060,-0.028,0.328,0.122,0.081,-0.155,-0.265,-0.153,-0.478,0.029,0.170,0.087,0.029,-0.366,-0.174,-0.291,-0.239,-0.180,-0.581,0.046,0.003,-0.354,-0.029,-0.166,-0.084,0.058,-0.120,-0.079,0.139,-0.214,-0.037,-0.045,-0.212,-0.155,-0.333,-0.082,0.252,-0.148,0.045,-0.079,0.187,0.196,-0.139,0.006,-0.027,0.027,-0.036,0.194,0.156,0.248,-0.834,-0.001,0.059,0.197,0.221,0.151,-0.045,0.067,-0.281,0.033,-0.121,0.039,0.022,-0.060,0.281,-0.071,0.414,0.036,0.056,-0.272,0.059,-0.049,-0.049,0.162,0.137,0.004,0.151,-0.178,-0.076,0.133,-0.052,0.023,0.003,0.002,0.098,-0.230,-0.052,0.425,-0.053,-0.065,0.156,0.034,-0.126,0.139,0.284,-0.364,0.153,-0.006,-0.012,-0.006,0.004,-0.090,-0.143,-0.111,-0.307,0.076,0.131,0.005,0.078,-0.153,-0.140,-0.021,-0.364,-0.559,-0.017,-0.344,-0.008,-0.196,-0.061,-0.239,0.036,-0.061,-0.143,-0.369,0.038,-0.021,-0.108,0.049,-0.080,-0.070,-0.008,-0.235,0.027,0.104,-0.089,-0.074,0.362,0.016,0.038,0.468,0.031,0.018,0.010,-0.008,0.018,0.110,-0.001,0.254,0.128,0.350,0.239,0.140,0.092,0.126,0.085,0.354,-0.077,0.013,0.324,0.155,-0.097,0.160,0.028,0.568,-0.186,0.163,0.016,-0.075,0.116,-0.079,0.012,0.804,-0.074,-0.173,-0.024,-0.052,0.310,0.104,0.121,-0.005,-0.290,-0.018,0.157,0.139,0.063,0.244,0.001,-0.498,-0.167,0.047,-0.005,-0.044,-0.121,-0.129,0.040,-0.452,-0.104,-0.155,-0.252,-0.269,0.175,-0.017,-0.082,-0.453,-0.007,-0.095,0.266,0.043,-0.087,0.097,0.198,-0.377,0.111,0.354,0.135,-0.121,0.053,-0.334,-0.054,-0.209,0.064,-0.140,-0.287,0.074,-0.066,0.116,0.003,0.103,0.005,0.177,0.021,0.047,-0.022,-0.298,-0.110,0.256,0.113,0.137,-0.079,-0.030,-0.218,-0.193,-0.112,0.313,-0.019,0.011,-0.206,-0.016,-0.133,-0.276,-0.089,0.222,-0.275,-0.044,0.024,-0.072,0.158,-0.000,-0.060,0.285,-0.475,0.065,-0.135,0.117,0.037,-0.056,0.031,0.235,-0.830,-0.082,-0.223,-0.066,0.239,0.054,0.046,0.121,-0.779,0.061,-0.174,-0.034,0.164,0.022,-0.066,0.427,-0.386,0.014,-0.277,0.271,0.098,-0.141,0.036,-0.008,-0.420,0.067,-0.389,0.052,0.175,0.133,-0.025,-0.758,-0.241,0.120,-0.200,0.036,0.001,-0.145,-0.206,-0.405,0.107,-0.196,-0.181,-0.271,-0.160,-0.102,-0.117,-0.067,0.026,0.010,0.052,0.009,0.016,0.050,-0.041,0.040,0.172,0.193,-0.159,-0.109,-0.530,-0.324,-0.057,0.202,0.084,0.089,-0.075,0.010,-0.474,-0.337,0.081,0.146,-0.014,0.009,0.026,0.053,-0.271,-0.182,0.005,0.130,-0.078,0.026,0.050,0.049,0.071,-0.104,0.052,0.217,-0.133,-0.022,-0.157,-0.006,0.162,-0.002,0.005,0.284,-0.552,0.004,0.110,0.129,0.246,-0.089,-0.215,0.154,-0.591,0.140,-0.000,-0.095,0.087,0.013,-0.134,0.061,-0.454,0.100,-0.097,-0.442,-0.041,-0.029,-0.087,-0.398,-0.318,-0.158,-0.078,-0.382,0.066,0.012,-0.023,-0.860,-0.070,-0.280,-0.290,-0.240,-0.268,-0.240,-0.066,-0.559,0.096,-0.150,-0.095,-0.056,0.071,-0.168,0.067,0.085,0.062,0.087,0.130,-0.185,-0.267,-0.184,0.012,0.005,0.182,0.030,-0.066,-0.004,-0.388,-0.365,-0.054,0.037,0.275,-0.010,-0.103,-0.148,-0.456,-0.371,-0.040,0.084,0.264,0.000,-0.031,-0.197,-0.093,-0.236,-0.044,0.211,0.069,-0.150,0.155,-0.059,-0.078,0.082,-0.026,0.263,0.196,-0.135,0.086,0.074,0.021,0.005,-0.231,0.141,0.046,-0.140,-0.087,0.069,-0.051,0.081,-0.164,0.054,-0.281,-0.128,-0.084,-0.161,-0.088,0.122,-0.215,-0.116,-0.346,-0.168,0.166,-0.152,0.054,-0.039,-0.190,-0.377,-0.265,-0.220,0.086,-0.255,0.118,-0.019,-0.151,-0.798,-0.101,-0.049,-0.101,-0.038,-0.548,-0.134,0.002,-0.347,0.074,-0.192,-0.019,-0.141,-0.476,-0.203,0.103,0.024,0.010,-0.226,-0.244,-0.227,-0.412,-0.478,0.225,0.096,0.058,0.257,-0.027,-0.091,-0.134,-0.359,0.249,0.069,0.161,0.100,-0.070,-0.043,-0.048,-0.453,-0.030,0.161,0.041,-0.048,-0.093,-0.265,-0.302,-0.153,0.095,0.220,0.094,-0.044,0.033,-0.178,-0.393,0.098,-0.051,0.265,0.165,-0.211,-0.329,0.038,-0.277,-0.033,-0.276,0.117,0.210,-0.145,-0.233,0.040,-0.107,0.026,-0.017,-0.005,0.107,-0.121,0.066,-0.103,-0.023,0.036,-0.056,0.010,-0.145,0.057,0.133,0.095,0.083,-0.024,-0.115,-0.394,-0.019,0.040,0.125,-0.051,-0.041,-0.049,-0.084,-0.187,-0.030,0.007,0.296,0.061,-0.384,-0.105,-0.190,0.026,0.025,-0.182,0.287,-0.245,0.051,-0.141,0.085,-0.063,0.020,-0.070,-0.184,-0.046,-0.173,-0.302,0.562,0.084,0.130,0.085,-0.065,-0.071,-0.098,-0.597,0.353,0.299,0.172,0.037,-0.120,-0.089,-0.180,-0.726,0.343,0.295,-0.034,0.109,-0.367,-0.019,-0.387,-0.532,0.212,0.092,-0.092,0.207,-0.403,0.172,-0.287,-0.102,0.162,0.054,0.033,-0.073,-0.335,0.184,-0.233,0.040,-0.082,-0.132,0.060,-0.232,-0.225,-0.017,-0.084,0.049,-0.003,-0.280,0.215,-0.099,0.084,0.018,-0.208,0.033,-0.049,-0.406,0.091,-0.051,0.041,-0.025,0.030,-0.014,-0.024,-0.285,0.107,0.038,0.293,0.182,-0.202,-0.166,0.028,-0.005,0.071,0.205,0.293,0.449,-0.336,-0.132,0.105,0.077,0.117,0.348,0.549,-0.058,0.287,-0.112,0.029,-0.379,0.046,0.023,0.051,0.114,-0.203,-0.312,0.269,-0.084,0.130,0.040,-0.075,0.110,-0.235,-0.597,0.216,-0.107,0.025,0.262,-0.004,0.068,-0.165,-0.724,0.154,-0.150,0.119,0.112,-0.147,0.041,-0.062,-0.282,0.109,-0.180,-0.116,0.253,-0.269,0.216,-0.276,-0.157,0.262,-0.317,-0.378,0.121,-0.102,-0.109,-0.162,-0.016,0.267,-0.348,0.301,-0.112,-0.153,0.028,-0.255,-0.056,0.055,-0.361,0.250,-0.005,0.020,0.054,-0.136,-0.022,-0.034,-0.299,0.216,0.039,0.144,0.161,-0.131,-0.169,-0.068,-0.031,0.045,0.051,0.096,0.030,-0.120,-0.143,-0.142,0.264,0.090,0.132,-0.199,0.333,-0.051,-0.233,0.174,0.399,0.113,0.262,0.186,-0.003,0.429,-0.112,0.064,-0.139,-0.012,0.056,0.460,0.033,0.249,-0.188,0.098,-0.073,0.103,-0.082,-0.033,0.128,0.147,-0.289,-0.053,-0.231,-0.110,0.081,0.121,-0.026,0.076,-0.300,0.006,-0.281,0.171,0.103,-0.027,0.072,0.074,-0.002,0.104,-0.240,0.143,0.212,0.155,0.103,-0.080,-0.003,0.151,-0.194,0.208,0.162,0.098,0.197,-0.336,0.123,0.048,-0.193,0.229,0.146,-0.014,-0.033,-0.187,-0.096,0.120,-0.078,0.032,0.034,0.030,-0.150,-0.104,-0.155,0.059,-0.009,0.088,0.095,-0.082,0.188,-0.077,-0.136,0.040,0.169,-0.016,0.200,-0.057,0.142,-0.072,-0.054,0.045,0.561,-0.160,0.152,-0.606,0.352,-0.157,-0.241,0.036,0.613,0.041,0.263,-0.052,0.132,0.024,-0.010,0.055,-0.023,0.049,-0.102,0.156,0.101,-0.161,-0.178,-0.205,0.038,0.006,0.109,0.004,-0.037,0.124,-0.114,-0.158,-0.059,0.082,-0.005,0.186,0.040,0.208,-0.011,-0.173,-0.161,0.015,-0.031,0.127,-0.026,0.106,0.027,0.038,-0.150,0.222,0.138,0.250,-0.027,-0.100,0.053,0.183,0.018,0.104,0.136,0.051,0.200,-0.120,-0.033,0.289,0.173,-0.057,0.267,0.049,0.178,0.015,0.004,0.134,0.150,-0.057,0.128,-0.016,0.165,0.023,-0.090,0.217,0.218,-0.106,0.002,0.013,-0.016,0.032,-0.081,0.079,0.481,-0.191,-0.027,-0.091,0.146,0.134,-0.056,-0.012,0.300,-0.017,-0.069,-0.274,0.218,-0.292,-0.191,-0.109,0.203,-0.011,0.086,-0.138,-0.063,0.023,-0.095,-0.011,-0.005,0.053,0.140,0.256,0.090,-0.158,0.217,-0.274,-0.160,0.393,-0.045,0.112,-0.020,0.127,0.237,-0.275,-0.113,0.386,-0.049,0.094,-0.080,0.150,0.075,-0.145,-0.072,0.366,-0.179,0.129,-0.192,0.095,-0.225,-0.086,-0.108,0.076,0.061,0.191,-0.080,0.101,-0.193,0.034,0.064,-0.009,0.092,0.084,0.184,0.006,-0.044,0.176,0.079,-0.094,0.254,-0.027,0.154,0.053,0.006,0.224,0.285,-0.257,0.005,0.004,0.099,0.274,-0.061,0.498,0.356,-0.115,0.001,-0.308,0.085,0.016,-0.015,0.416,0.287,-0.158,-0.186,-0.190,0.266,0.103,-0.136,0.232,0.350,-0.124,-0.017,-0.214,0.355,-0.027,-0.028,0.091,0.062,-0.085,0.188,-0.110,0.153,-0.158,0.044,-0.783,0.097,0.084,0.119,0.370,-0.123,-0.110,0.078,-0.586,0.078,0.374,0.017,-0.021,-0.029,0.054,-0.146,-0.560,0.037,0.164,0.024,0.085,-0.065,0.121,0.065,-0.309,0.085,0.036,-0.075,0.225,-0.130,0.173,-0.207,-0.082,0.079,-0.012,-0.133,0.080,-0.129,-0.006,-0.262,-0.007,0.061,0.010,0.083,-0.217,-0.049,0.256,-0.112,0.096,0.078,-0.172,-0.026,-0.059,0.055,0.038,0.141,0.346,0.133,-0.109,-0.041,-0.085,0.082,-0.081,0.019,0.474,0.348,0.023,-0.087,-0.118,0.013,0.007,-0.060,0.557,0.178,0.107,-0.450,-0.229,-0.072,-0.046,-0.172,0.290,0.175,-0.010,-0.096,-0.255,0.317,0.158,-0.053,0.092,0.096,0.005,0.219,0.109,-0.264,0.062,0.001,-0.840,-0.136,0.244,-0.405,-0.034,0.030,-0.230,-0.191,-0.268,0.002,-0.014,-0.026,-0.187,-0.060,-0.382,-0.575,-0.977,0.115,0.120,0.110,0.121,-0.154,-0.232,-0.280,-0.361,0.199,-0.133,-0.031,-0.061,-0.326,-0.187,-0.444,-0.379,0.087,-0.082,0.029,-0.117,-0.134,-0.257,-0.446,-0.153,0.063,-0.045,0.138,-0.155,-0.470,-0.259,-0.600,-0.032,0.095,-0.156,-0.254,-0.397,-0.170,-0.107,-0.148,0.260,0.022,-0.131,-0.155,-0.480,-0.058,-0.259,-0.076,0.596,0.055,0.071,-0.236,-0.224,0.093,-0.305,-0.347,0.262,0.008,-0.023,-0.245,-0.084,0.143,-0.287,-0.297,0.072,-0.064,-0.194,-0.190,-0.063,-0.075,0.028,-0.131,-0.115,-0.013,-0.046,0.094,0.094,-0.101,0.015,0.056,-0.030,-0.143,-0.082,0.054,0.019,-0.105,-0.079,0.183,-0.045,-0.039,-0.082,0.004,0.270,0.192,-0.009,-0.227,-0.124,-0.224,-0.166,0.026,0.082,0.272,-0.013,-0.044,-0.060,0.120,-0.207,-0.029,0.132,-0.287,0.067,-0.198,-0.002,0.094,-0.003,-0.064,-0.057,-0.198,-0.152,0.076,0.057,0.400,-0.087,-0.030,-0.259,-0.091,-0.021,-0.110,0.060,0.233,-0.100,0.013,-0.109,0.034,-0.048,0.007,0.072,0.165,-0.188,0.063,0.041,-0.038,0.180,-0.157,0.052,-0.333,-0.306,-0.071,-0.049,-0.018,0.131,0.152,0.186,-0.325,-0.332,0.140,0.133,-0.015,-0.146,0.245,0.233,-0.244,-0.169,0.135,-0.321,-0.243,-0.106,0.213,-0.200,-0.199,-0.187,-0.722,-0.238,0.091,0.174,0.228,0.044,0.077,-0.181,0.382,0.337,0.258,-0.239,0.102,-0.042,0.057,-0.115,0.126,0.165,-0.136,0.106,0.028,0.044,0.127,-0.118,-0.104,0.035,-0.037,0.177,-0.094,0.136,0.407,-0.392,0.179,-0.105,0.191,0.193,-0.030,0.122,0.293,-0.522,0.130,0.090,0.163,0.175,0.074,0.067,0.380,-0.586,-0.181,-0.347,-0.054,0.157,-0.154,0.044,0.301,-0.495,0.034,-0.065,-0.127,0.034,0.052,-0.031,0.153,-0.204,-0.108,-0.062,-0.216,-0.063,-0.141,0.042,0.016,-0.318,-0.198,-0.200,-0.085,0.171,0.046,0.076,-0.514,-0.190,-0.061,-0.452,-0.206,0.279,0.040,0.164,-0.443,-0.213,0.003,-0.453,0.014,0.192,0.236,0.136,-0.244,-0.114,0.210,-0.552,0.110,-0.369,-0.292,0.085,0.145,-0.220,0.329,0.301,0.122,0.079,-0.439,0.041,0.200,-0.060,0.032,-0.108,0.085,-0.035,-0.140,0.001,0.080,-0.061,-0.057,-0.191,0.082,0.128,-0.219,0.102,0.160,-0.251,0.152,-0.137,0.268,0.058,-0.090,0.061,0.241,-0.379,-0.028,-0.125,0.122,0.023,-0.152,-0.018,0.121,-0.473,0.061,-0.218,0.146,-0.165,0.023,-0.059,0.078,-0.343,0.015,-0.208,-0.316,-0.068,-0.013,-0.107,-0.167,-0.009,-0.162,-0.339,-0.331,-0.029,0.012,-0.121,-0.126,-0.055,-0.138,-0.285,-0.150,0.078,0.092,-0.118,-0.400,-0.030,-0.278,-0.372,-0.355,0.136,0.013,-0.006,-0.493,-0.272,-0.330,-0.695,-0.227,0.192,0.089,-0.111,-0.446,-0.004,-0.227,-1.140,0.024,0.055,-0.301,-0.056,0.066,-0.172,0.068,0.012,0.050,-0.223,-0.098,0.120,0.124,0.138,-0.063,0.186,0.159,-0.260,-0.538,-0.042,0.244,0.025,0.155,-0.005,0.067,-0.206,-0.655,0.071,0.163,0.024,0.153,-0.035,0.104,-0.448,-0.609,-0.072,0.080,-0.255,0.185,0.016,0.074,-0.291,-0.280,-0.012,0.150,0.097,0.079,-0.027,0.131,-0.192,-0.156,-0.138,-0.010,0.059,0.021,0.013,0.144,-0.014,-0.025,-0.048,-0.085,0.175,0.032,0.019,0.189,-0.072,0.045,-0.158,-0.229,0.154,-0.084,-0.079,-0.279,0.041,0.061,-0.084,-0.359,0.062,-0.269,-0.081,-0.126,0.126,0.025,-0.073,-0.529,0.080,-0.059,0.115,-0.060,0.005,0.140,-0.117,-0.395,-0.014,-0.390,-0.670,0.005,0.122,0.130,-0.164,-0.000,-0.037,0.245,0.046,0.048,-0.169,-0.130,0.015,0.108,0.116,-0.019,-0.107,-0.050,-0.405,-0.680,-0.171,0.220,0.105,0.210,-0.052,0.039,-0.590,-0.620,-0.185,0.127,0.190,0.209,-0.061,0.231,-0.468,-0.574,-0.158,0.059,0.189,0.272,-0.161,0.128,-0.278,-0.474,0.032,0.096,0.128,0.211,0.065,0.061,-0.088,-0.184,0.057,-0.056,-0.110,0.370,0.050,0.241,-0.055,-0.018,0.132,-0.017,-0.020,0.285,0.034,0.106,-0.011,0.039,-0.095,-0.073,0.154,0.245,-0.105,-0.048,0.030,-0.084,-0.223,-0.225,0.212,-0.097,0.001,-0.441,0.204,-0.009,-0.264,-0.023,0.160,-0.313,0.117,-0.910,-0.521,0.098,-0.190,-0.252,-0.040,-0.698,-0.704,-0.258,0.209,-0.363,0.104,-0.061,-0.107,0.044,0.143,-0.109,-0.210,-0.257,0.049,0.023,0.393,-0.074,-0.214,-0.180,-0.321,-0.290,-0.025,0.099,0.221,0.118,-0.199,0.049,-0.513,-0.354,-0.257,-0.037,0.311,0.173,0.029,0.252,-0.376,-0.438,0.021,0.094,0.200,0.276,0.086,0.197,-0.158,-0.785,0.224,0.126,-0.054,0.416,0.332,0.220,0.124,-0.238,0.259,0.149,-0.044,0.379,0.187,0.148,0.043,-0.174,0.242,0.230,-0.167,0.352,-0.063,0.238,-0.055,-0.013,0.020,0.187,0.113,0.335,-0.148,0.090,0.029,-0.043,0.022,-0.050,0.037,0.208,-0.290,-0.081,0.002,-0.049,0.037,0.181,0.047,0.050,-0.662,-0.356,-0.135,-0.071,-0.108,0.034,-0.150,-0.256,-0.488,-0.109,-0.013,0.302,0.120,-0.219,-0.216,-0.089,0.154,-0.254,-0.184,-0.021,-0.180,-0.271,0.228,-0.425,-0.346,-0.175,-0.114,0.192,0.065,0.031,-0.158,-0.293,-0.258,-0.233,-0.113,-0.116,-0.067,0.142,-0.082,0.005,-0.009,0.018,-0.122,-0.419,0.043,0.124,-0.228,-0.100,0.205,-0.033,0.059,-0.409,0.062,0.247,-0.221,-0.013,0.261,0.049,0.211,-0.095,0.180,0.337,-0.221,0.298,0.055,0.251,0.159,-0.179,0.166,0.101,-0.102,0.183,-0.135,0.089,-0.017,-0.100,-0.033,0.087,-0.081,-0.124,-0.310,-0.108,0.058,0.008,0.082,-0.122,-0.081,-0.460,-0.027,-0.297,0.024,0.015,0.085,-0.395,-0.176,-0.179,-0.555,0.088,0.212,0.043,-0.012,-0.327,-0.087,-0.099,-0.434,-0.137,0.107,0.192,0.031,-0.275,0.039,-0.151,-0.039,-0.198,0.114,0.112,0.030,-0.147,-0.279,-0.106,-0.320,-0.303,0.097,-0.053,0.154,0.197,-0.462,-0.284,-0.347,-0.197,-0.140,-0.204,0.029,0.170,-0.242,-0.256,-0.133,0.024,-0.065,-0.074,0.091,0.274,-0.347,-0.163,-0.039,-0.065,0.251,-0.141,0.007,0.288,-0.239,-0.050,0.194,-0.130,0.241,0.051,0.010,0.177,-0.003,0.122,-0.090,-0.306,0.069,-0.058,0.050,0.089,-0.078,-0.017,-0.293,-0.016,0.000,-0.053,0.074,0.029,-0.041,-0.024,-0.062,-0.249,-0.083,-0.006,-0.003,-0.134,-0.143,-0.006,-0.014,-0.200,-0.042,0.036,-0.074,-0.451,-0.085,-0.082,-0.041,-0.074,-0.035,0.028,0.056,-0.431,-0.052,-0.256,-0.434,-0.258,0.182,-0.047,-0.094,0.104,-0.127,-0.055,0.078,-0.085,0.247,0.100,0.330,0.031,-0.268,-0.257,-0.097,-0.009,0.102,-0.108,0.147,0.076,-0.503,-0.131,-0.159,-0.175,-0.090,-0.187,0.046,0.055,-0.275,-0.222,-0.157,0.135,-0.109,-0.130,0.129,0.298,-0.323,-0.090,-0.267,0.129,0.054,-0.369,0.235,0.304,-0.139,-0.009,0.080,-0.121,0.134,-0.158,0.103,0.058,-0.153,-0.094,-0.369,0.175,-0.049,-0.093,0.032,0.089,0.042,-0.009,-0.061,-0.093,-0.013,0.066,-0.019,-0.088,0.017,-0.095,0.103,-0.309,-0.003,0.046,-0.080,-0.240,0.028,-0.090,-0.124,-0.207,0.088,0.151,-0.068,-0.389,-0.047,-0.244,0.138,-0.216,0.048,0.097,0.079,-0.071,0.045,-0.284,-0.069,0.077,0.439,0.261,0.147,0.129,-0.260,0.158,0.018,-0.157,0.048,0.151,0.258,0.206,-0.434,-0.140,-0.072,-0.307,0.115,-0.250,0.110,0.133,-0.502,-0.064,-0.151,-0.011,-0.008,-0.180,0.143,0.146,-0.440,0.061,-0.124,0.237,0.072,-0.387,0.179,0.095,-0.281,0.114,-0.226,0.123,-0.178,-0.487,0.049,0.129,-0.061,-0.021,-0.334,0.147,-0.185,-0.105,0.008,0.026,0.012,-0.034,-0.190,-0.042,0.016,-0.042,0.079,-0.117,0.207,-0.011,-0.127,0.105,-0.136,0.059,-0.123,-0.287,0.100,0.015,0.029,0.099,-0.215,-0.060,-0.129,-0.277,-0.015,0.019,0.152,-0.152,-0.025,-0.006,-0.128,-0.307,0.072,-0.033,0.097,-0.163,-0.080,0.044,-0.114,-0.131,0.050,-0.034,-0.521,0.032,0.135,0.022,0.681,-0.026,-0.273,0.008,0.049,0.017,-0.051,-0.035,0.717,0.084,-0.513,-0.117,-0.103,0.055,-0.053,-0.154,0.338,0.042,-0.428,0.049,-0.052,0.094,-0.052,-0.335,0.296,0.002,-0.374,-0.004,-0.350,0.085,-0.232,-0.613,0.194,0.022,-0.085,0.130,-0.297,0.183,-0.264,-0.323,0.094,-0.053,-0.034,0.156,-0.157,0.096,-0.228,-0.104,0.196,-0.082,0.207,0.130,-0.086,0.307,-0.139,-0.084,0.182,-0.119,0.039,0.107,0.035,0.105,0.050,0.046,-0.073,-0.222,0.039,-0.002,0.114,0.057,-0.228,-0.025,-0.564,-0.109,-0.004,0.163,-0.114,0.005,-0.092,-0.055,-0.628,-0.078,0.031,0.025,0.135,-0.454,-0.087,0.153,-0.366,-0.122,0.081,-0.342,-0.589,0.008,-0.074,0.013,0.149,0.184,-0.230,0.267,0.158,-0.028,0.267,0.175,0.415,0.060,-0.189,0.172,0.081,0.031,0.120,-0.231,0.424,0.080,-0.218,0.111,-0.019,0.273,-0.171,-0.220,0.568,0.098,-0.115,0.150,-0.091,0.216,0.059,-0.213,0.669,-0.005,-0.139,0.128,0.026,0.154,-0.263,-0.114,0.062,-0.074,0.086,0.196,0.059,0.291,-0.329,-0.171,-0.311,-0.131,0.054,0.163,0.106,0.296,-0.058,-0.059,-0.341,-0.071,0.012,0.322,0.065,0.291,0.156,0.201,-0.351,-0.056,0.089,0.158,0.192,0.387,-0.053,0.125,-0.112,-0.058,0.005,0.389,-0.008,0.027,-0.087,-0.023,-0.305,-0.134,-0.019,0.133,-0.228,-0.139,-0.052,-0.033,-0.142,-0.055,-0.019,-0.064,-0.027,-0.201,0.132,-0.073,-0.229,-0.034,-0.108,-0.092,-0.159,-0.277,-0.218,-0.137,-0.086,-0.279,-0.108,-0.096,0.038,-0.452,-0.156,-0.445,-0.149,-0.285,-0.095,-0.286,-0.261,-0.222,-0.491,-0.162,-0.387,-0.284,-0.191,-0.337,-0.288,-0.256,-0.441,-0.359,-0.208,-0.755,-0.315,-0.421,-0.454,-0.273,-0.246,-0.295,-0.071,-0.591,-0.201,-0.416,-0.353,-0.158,-0.250,-0.116,-0.106,-0.639,-0.195,-0.219,-0.134,0.108,-0.294,-0.122,-0.084,-0.664,-0.043,0.010,-0.138,-0.001,-0.180,-0.378,-0.062,-0.376,-0.168,-0.421,-0.373,-0.187,0.133,-0.064,-0.016,-0.263,-0.322,-0.462,-0.042,0.083,0.070,0.150,0.089,-0.129,-0.213,0.024,-0.067,0.091,0.124,0.095,-0.059,-0.142,-0.124,-0.195,-0.128,-0.260,-0.144,-0.101,0.137,-0.240,-0.066,-0.101,-0.068,-0.366,0.017,-0.091,-0.047,-0.271,-0.194,-0.075,-0.240,-0.212,-0.083,-0.056,-0.026,-0.167,-0.279,0.040,-0.068,-0.069,-0.054,0.254,-0.190,-0.292,-0.159,-0.183,-0.100,0.183,0.022,0.271,-0.193,0.064,-0.126,-0.163,-0.309,-0.123,-0.179,0.221,-0.298,-0.159,-0.227,-0.344,0.055,0.139,-0.112,0.091,-0.210,-0.535,-0.121,-0.150,0.038,-0.026,-0.093,0.051,-0.102,0.086,0.192,-0.138,-0.116,0.088,0.006,-0.056,-0.065,-0.021,0.041,-0.096,0.033,0.152,-0.128,0.070,-0.110,-0.314,-0.050,0.078,0.213,0.092,0.121,-0.046,0.072,-0.041,0.022,0.131,-0.184,0.196,-0.211,0.038,0.224,-0.129,-0.032,0.077,0.048,-0.085,0.067,0.040,0.069,0.011,-0.056,-0.317,-0.204,-0.185,0.040,-0.179,0.011,-0.169,-0.031,-0.270,-0.224,0.297,0.171,0.152,-0.097,0.017,-0.360,-0.032,-0.035,-0.137,0.323,0.383,-0.190,-0.087,0.116,-0.373,0.127,-0.163,0.332,0.419,-0.231,-0.189,0.311,-0.497,0.129,-0.264,0.203,0.674,-0.252,-0.567,0.231,-0.493,0.357,-0.212,0.073,0.433,-0.276,-0.924,0.428,-0.497,0.309,-0.174,-0.189,0.327,-0.164,-0.708,0.329,-0.222,0.562,0.053,-0.259,0.188,-0.066,-0.582,0.364,-0.147,0.447,0.163,0.020,0.083,-0.101,-0.338,0.128,-0.029,0.447,0.180,-0.124,0.034,0.013,0.163,0.159,0.083,0.216,0.067,-0.031,0.012,0.173,0.262,-0.089,0.153,0.256,0.104,0.118,-0.022,-0.062,0.115,-0.147,-0.019,-0.167,0.087,0.180,-0.107,-0.110,-0.001,-0.146,-0.440,-0.001,-0.145,0.209,0.259,0.069,0.019,-0.018,-0.304,-0.063,-0.116,0.344,0.487,-0.003,-0.363,0.038,-0.375,0.026,-0.070,0.158,0.407,-0.177,-0.518,0.165,-0.205,0.058,-0.177,0.138,0.253,-0.397,-0.607,0.084,-0.195,0.116,-0.049,-0.109,0.240,-0.183,-0.356,0.226,-0.195,0.405,-0.166,0.122,0.267,-0.130,-0.594,0.142,-0.151,0.431,0.275,0.009,0.220,-0.058,-0.311,0.235,-0.103,0.335,0.231,0.039,0.058,-0.069,-0.219,0.139,0.159,0.389,0.214,-0.265,0.041,-0.037,0.163,0.060,0.228,0.186,0.022,0.190,-0.131,-0.152,0.397,0.092,0.146,0.292,0.119,-0.183,0.095,0.085,0.006,0.002,0.013,0.090,-0.156,-0.104,-0.161,0.027,-0.221,-0.064,-0.271,-0.088,0.060,0.113,0.169,0.046,-0.156,0.036,-0.031,-0.029,0.238,0.338,0.201,0.134,-0.256,-0.008,-0.074,0.133,0.183,0.286,0.322,0.047,-0.307,-0.048,-0.058,0.215,0.200,0.055,0.142,0.027,-0.536,0.081,0.015,0.220,-0.153,-0.024,0.011,-0.160,-0.136,0.088,-0.257,0.211,0.061,0.160,0.118,-0.135,-0.051,-0.056,-0.176,0.382,0.060,0.104,0.143,0.007,-0.159,0.079,0.044,0.331,-0.043,0.114,0.042,-0.062,-0.271,0.125,-0.001,0.273,-0.006,-0.242,0.076,-0.209,0.161,0.110,0.005,-0.033,0.004,-0.114,-0.280,-0.376,0.189,-0.082,-0.017,0.028,-0.042,0.042,0.268,0.074,-0.016,-0.026,-0.385,-0.101,-0.190,0.059,-0.001,-0.146,-0.051,-0.346,-0.024,0.080,0.184,0.011,0.298,0.097,-0.181,-0.451,0.149,0.037,0.102,0.136,0.217,-0.051,-0.289,-0.155,0.095,0.049,0.125,0.070,0.221,0.128,-0.086,-0.166,0.143,0.062,0.034,0.012,0.128,-0.054,-0.020,-0.022,0.101,0.234,-0.065,-0.079,-0.120,-0.119,-0.309,-0.161,-0.222,-0.146,-0.224,0.135,0.209,-0.069,-0.546,-0.027,-0.216,0.188,-0.026,0.199,0.053,-0.162,-0.102,-0.027,0.045,0.161,0.173,-0.018,0.003,-0.097,0.011,-0.062,0.008,0.229,0.189,0.031,0.044,-0.022,0.086,0.037,0.065,-0.077,0.135,-0.270,-0.174,-0.033,0.030,-0.135,0.016,-0.436,-0.043,-0.297,-0.052,-0.176,0.148,-0.133,0.012,-0.386,0.138,0.041,0.086,-0.040,0.121,-0.723,0.187,0.184,0.008,0.029,0.038,-0.015,0.046,-0.646,0.079,-0.123,0.063,0.027,0.132,0.038,0.028,-0.270,0.215,0.098,-0.083,-0.028,-0.100,0.001,0.273,-0.059,-0.053,-0.027,-0.004,-0.093,0.024,-0.055,0.158,-0.254,-0.173,-0.222,-0.127,-0.121,-0.241,-0.146,-0.059,-0.188,-0.135,-0.418,-0.152,-0.010,0.040,0.054,-0.136,0.052,0.000,-0.062,0.048,0.024,-0.012,0.007,0.203,0.019,0.189,0.373,0.165,-0.095,0.023,-0.066,0.353,0.032,0.118,0.215,0.068,-0.102,-0.058,0.061,0.288,0.025,-0.023,-0.278,-0.072,-0.375,-0.191,0.032,0.155,-0.019,-0.074,-0.400,0.105,-0.205,-0.152,0.088,0.168,-0.253,0.217,0.059,-0.035,0.068,-0.185,0.056,0.036,-0.522,-0.044,0.024,-0.079,-0.088,-0.267,-0.131,0.142,-0.197,0.167,-0.013,-0.089,0.095,0.097,-0.140,0.066,-0.207,-0.000,0.133,-0.432,-0.275,-0.024,0.033,0.072,-0.305,-0.163,-0.055,-0.106,-0.135,0.207,0.135,0.173,-0.340,0.018,-0.094,-0.125,-0.292,-0.348,0.066,0.119,0.155,0.015,-0.179,-0.062,-0.068,-0.162,-0.098,0.075,0.086,0.055,0.073,-0.235,-0.053,-0.075,-0.135,0.020,-0.020,0.222,0.010,0.169,0.033,0.019,-0.075,0.102,0.000,0.165,0.070,0.239,-0.129,0.002,0.151,-0.073,0.016,-0.015,-0.091,0.007,-0.344,-0.022,-0.093,-0.087,-0.014,0.054,-0.303,0.103,-0.232,-0.223,0.194,0.059,-0.134,0.093,0.177,0.166,-0.086,-0.141,0.036,0.048,0.062,0.147,-0.021,0.080,-0.069,-0.254,-0.006,0.013,-0.099,0.113,-0.021,-0.175,-0.196,0.029,0.034,-0.021,0.049,0.075,-0.144,-0.223,-0.396,-0.079,-0.115,-0.019,-0.030,0.069,-0.094,-0.086,-0.129,-0.098,-0.061,-0.065,-0.163,0.112,-0.038,-0.039,-0.149,-0.105,0.035,-0.052,-0.096,0.011,0.101,-0.087,0.016,-0.065,-0.177,-0.108,-0.157,0.190,0.074,-0.207,0.108,-0.099,-0.263,0.183,-0.066,0.117,-0.059,0.116,0.123,0.031,-0.167,0.181,0.094,0.122,0.096,0.114,-0.071,0.138,-0.123,0.056,-0.021,0.139,-0.069,0.203,-0.206,-0.019,-0.043,-0.038,-0.113,0.009,-0.396,0.053,-0.120,-0.099,-0.259,0.041,-0.038,0.023,0.131,0.049,-0.304,-0.209,-0.099,0.106,-0.114,0.276,-0.011,0.092,-0.477,-0.537,0.118,0.044,-0.025,0.087,-0.008,0.130,-0.407,-0.019,0.031,-0.151,0.244,0.114,0.017,-0.106,-0.350,0.012,-0.235,-0.183,0.056,0.093,0.004,-0.080,-0.473,0.090,-0.513,-0.129,0.119,-0.026,0.117,-0.098,-0.047,-0.016,-0.627,-0.064,-0.002,0.003,0.126,-0.312,0.106,0.082,-0.159,0.067,-0.054,-0.042,0.139,-0.340,0.159,0.062,-0.173,0.137,-0.049,-0.390,0.018,-0.292,0.159,-0.040,0.107,0.227,-0.143,-0.168,-0.152,-0.106,-0.005,0.149,0.009,0.010,0.082,-0.172,-0.160,0.296,-0.211,-0.072,-0.130,-0.098,0.033,-0.022,-0.458,-0.033,-0.045,-0.112,-0.003,-0.057,-0.148,-0.484,0.011,-0.135,-0.346,-0.344,0.109,0.018,-0.163,-0.374,-0.105,-0.039,0.049,-0.103,-0.164,-0.112,0.114,-0.119,0.031,-0.036,-0.110,0.090,-0.197,-0.167,0.220,-0.171,0.063,-0.099,-0.048,0.222,-0.340,-0.309,-0.024,-0.002,0.097,-0.049,-0.102,0.153,-0.250,-0.202,0.075,-0.151,0.107,0.226,0.071,0.189,-0.350,-0.074,-0.001,-0.103,0.204,-0.419,0.261,0.164,0.023,0.001,0.087,-0.492,0.263,-0.125,0.091,0.098,0.223,0.227,0.031,-0.175,0.247,0.116,-0.008,-0.073,0.075,0.168,0.049,-0.165,0.250,0.053,0.161,0.206,0.238,0.111,0.050,-0.047,-0.067,0.088,-0.077,-0.125,0.089,-0.079,-0.093,-0.171,-0.240,-0.555,-0.087,-0.009,0.147,-0.285,-0.120,-0.569,-0.250,-0.061,0.058,-0.196,-0.049,-0.309,-0.182,0.177,-0.447,0.041,-0.016,0.158,-1.005,-0.213,0.098,0.002,-0.038,-0.082,-0.030,0.040,-0.985,-0.154,-0.025,-0.041,0.117,-0.059,0.121,0.275,-0.430,-0.163,0.114,-0.056,0.035,-0.032,0.022,0.105,-0.852,-0.099,-0.004,0.002,0.107,0.003,0.142,0.055,-0.094,-0.072,-0.002,-0.179,0.133,-0.026,-0.070,-0.033,0.087,-0.063,-0.049,-0.055,0.205,0.284,0.017,-0.058,-0.036,-0.009,0.044,0.293,0.295,-0.055,-0.256,-0.284,-0.075,0.012,0.047,-0.217,0.208,-0.245,-0.057,-0.043,-0.130,0.052,-0.020,0.109,-0.015,0.302,-0.191,-0.088,0.090,0.015,-0.007,-0.047,-0.152,-0.037,-0.105,-0.072,0.125,-0.140,0.093,0.066,-0.139,-0.102,-0.140,-0.318,-0.298,-0.234,0.207,0.040,-0.166,0.160,-0.336,-0.290,-0.123,-0.304,0.254,-0.059,-0.188,-0.201,-0.142,-0.577,-0.074,-0.465,0.141,-0.369,-0.458,0.021,0.142,-0.173,-0.051,-0.683,0.224,-0.084,-0.020,0.172,0.048,-0.173,-0.138,-0.679,0.054,-0.219,-0.065,-0.193,-0.093,0.110,-0.009,-0.634,0.097,-0.293,0.063,-0.111,-0.125,-0.136,-0.053,-0.367,0.060,-0.298,-0.186,-0.358,0.183,0.028,-0.182,-0.159,0.016,-0.101,0.092,0.047,0.009,-0.121,-0.164,-0.258,0.176,0.071,-0.113,0.071,-0.135,-0.182,-0.088,-0.092,0.195,-0.093,-0.189,-0.138,-0.050,-0.130,-0.074,-0.191,0.131,-0.187,0.004,0.141,-0.352,0.005,0.109,-0.028,0.165,-0.110,0.014,-0.067,0.128,-0.678,-0.074,-0.212,0.253,-0.343,-0.142,-0.305,-0.097,-0.137,-0.081,-0.491,0.119,-0.432,-0.098,-0.158,0.076,-0.149,-0.126,-0.613,0.257,-0.173,0.024,-0.019,0.178,-0.064,-0.044,-0.537,0.361,0.077,0.104,-0.134,-0.034,-0.204,-0.015,-0.155,0.343,0.084,-0.069,-0.134,-0.212,-0.059,-0.130,0.128,0.215,-0.031,0.282,0.082,-0.073,-0.138,-0.038,-0.040,0.430,0.107,0.145,-0.127,-0.073,-0.077,-0.105,-0.034,0.416,-0.086,0.076,-0.267,-0.259,-0.111,-0.065,0.202,0.490,0.023,-0.192,0.079,-0.503,-0.307,-0.055,0.245,0.279,-0.061,0.129,0.089,-0.102,-0.182,-0.075,0.246,0.066,-0.067,0.117,0.059,-0.435,0.052,-0.283,-0.034,0.261,-0.394,-0.322,-0.090,0.047,-0.578,-0.052,-0.337,0.283,-0.241,0.016,-0.155,0.411,0.074,-0.079,-0.354,0.285,-0.145,-0.115,-0.146,0.264,-0.006,-0.182,-0.133,0.106,-0.144,0.088,-0.286,0.176,0.085,-0.081,-0.093,0.300,-0.030,-0.057,-0.107,0.119,-0.060,0.042,-0.104,0.146,0.023,0.043,-0.121,-0.214,-0.206,0.034,0.081,0.134,0.037,-0.008,-0.005,-0.021,-0.171,0.075,0.229,0.264,0.199,0.117,0.124,-0.119,-0.164,0.067,0.203,0.279,0.298,-0.050,0.311,-0.164,-0.210,0.067,0.285,0.389,0.318,-0.091,0.257,-0.508,-0.432,0.035,0.269,0.294,0.193,0.194,0.167,-0.155,-0.082,-0.017,0.318,-0.015,0.225,0.429,-0.120,-0.300,-0.196,-0.204,-0.182,0.272,-0.275,-0.156,-0.005,0.128,0.122,-0.316,-0.139,0.126,-0.156,-0.088,-0.104,0.168,0.199,-0.328,-0.089,-0.028,-0.226,0.001,-0.075,0.208,0.088,-0.265,-0.088,-0.373,-0.190,-0.079,-0.102,0.130,0.146,-0.237,-0.162,-0.010,-0.083,-0.166,-0.183,0.178,0.059,-0.079,0.021,0.014,-0.127,-0.103,-0.333,-0.010,-0.331,-0.009,0.263,-0.081,0.079,-0.213,-0.130,0.072,-0.363,0.243,0.366,-0.245,0.088,-0.279,-0.045,-0.057,-0.244,0.148,0.311,0.078,0.287,-0.446,0.213,-0.155,-0.303,0.042,0.392,0.213,0.375,-0.147,0.242,-0.218,-0.388,-0.026,0.356,0.013,0.116,-0.320,0.195,0.102,-0.073,0.023,0.200,0.038,0.194,0.101,-0.150,-0.399,-0.354,-0.343,-0.133,0.258,-0.295,-0.280,0.111,0.137,0.269,-0.248,-0.085,-0.022,-0.012,-0.131,0.099,0.085,0.175,-0.299,0.022,-0.494,-0.256,0.014,-0.135,0.049,0.322,-0.060,-0.070,-0.195,-0.096,-0.110,0.004,0.006,0.135,-0.067,0.075,-0.036,-0.186,-0.050,-0.137,0.089,0.100,0.045,0.069,-0.060,-0.162,0.006,-0.340,-0.238,-0.254,0.059,0.210,-0.178,-0.012,-0.131,-0.466,-0.153,-0.284,0.118,0.366,-0.453,-0.177,-0.555,-0.366,-0.029,-0.231,0.258,0.367,-0.297,-0.031,-0.858,-0.129,-0.322,-0.254,0.060,0.332,-0.322,0.207,-0.558,-0.025,-0.187,-0.528,0.002,0.330,-0.154,0.219,-0.482,0.060,0.101,0.003,-0.060,0.535,0.111,0.325,-0.218,-0.059,-0.389,-0.265,-0.455,-0.114,0.219,-0.062,-0.390,-0.076,-0.126,-0.301,-0.302,0.036,0.073,-0.044,-0.344,-0.266,0.005,-0.042,-0.092,0.278,-0.305,-0.099,-0.052,-0.124,0.022,0.076,-0.006,0.195,-0.171,-0.059,0.059,-0.159,0.096,0.031,-0.068,0.123,0.009,-0.017,0.203,0.001,0.025,0.163,0.033,0.117,0.081,-0.148,0.063,0.095,-0.206,0.015,0.103,0.059,0.026,-0.078,-0.206,-0.271,-0.125,-0.058,0.018,0.227,-0.243,-0.266,-0.640,-0.582,0.005,-0.064,0.130,0.283,-0.414,-0.374,-0.574,-0.187,-0.157,-0.038,0.187,0.172,-0.271,-0.261,-0.472,-0.125,-0.109,-0.215,0.050,0.124,-0.331,-0.468,-0.553,0.058,-0.035,-0.072,-0.027,0.212,0.028,-0.067,-0.468,-0.364,0.038,-0.339,0.058,-0.067,0.161,-0.089,-0.312,0.118,0.071,-0.027,0.135,0.305,-0.071,-0.033,-0.190,-0.027,0.060,0.004,0.171,0.541,0.046,-0.055,0.040,-0.065,0.038,-0.206,-0.104,0.217,-0.298,-0.007,-0.055,-0.095,0.041,-0.073,-0.033,0.084,-0.416,-0.010,0.056,-0.005,-0.004,-0.040,-0.030,0.068,-0.089,-0.065,-0.052,-0.106,-0.038,0.056,-0.110,0.167,0.025,-0.066,0.149,-0.215,0.015,0.094,0.010,0.119,-0.297,-0.383,-0.111,-0.187,0.063,-0.038,0.126,0.092,-0.459,-0.294,-0.045,-0.396,-0.090,-0.132,0.010,0.018,-0.318,-0.228,-0.166,0.028,-0.085,-0.175,0.194,-0.119,-0.225,-0.206,-0.022,0.021,-0.113,-0.025,-0.056,-0.076,-0.072,-0.119,-0.212,-0.029,-0.123,0.067,0.307,0.010,0.132,0.077,-0.233,-0.103,-0.043,-0.089,0.387,0.075,0.156,-0.093,-0.211,-0.041,-0.122,-0.028,0.338,0.220,0.083,-0.222,-0.172,-0.008,-0.098,-0.258,0.281,0.252,-0.149,-0.099,-0.193,-0.264,-0.218,-0.181,0.153,0.063,-0.225,-0.170,-0.052,0.012,-0.259,-0.068,0.050,0.040,0.008,-0.207,-0.132,-0.091,-0.016,-0.003,-0.041,-0.070,-0.156,-0.095,0.093,-0.032,0.040,0.104,-0.028,0.047,-0.128,-0.297,0.090,0.003,0.096,0.012,0.039,0.008,-0.219,-0.386,0.049,-0.039,0.008,-0.061,0.009,-0.186,-0.225,-0.142,-0.057,0.099,0.024,-0.039,-0.008,-0.160,-0.164,0.028,0.190,-0.171,-0.039,0.033,0.005,-0.089,-0.019,-0.166,0.052,0.206,0.007,0.292,0.061,0.072,0.190,0.159,0.036,-0.058,0.044,-0.076,0.185,0.140,0.054,0.068,-0.132,-0.112,-0.119,-0.076,0.243,0.206,0.335,-0.345,-0.159,-0.251,-0.144,-0.261,0.323,0.167,0.063,-0.283,-0.208,-0.190,-0.135,-0.103,0.187,0.050,0.060,-0.281,-0.182,-0.029,-0.075,-0.013,0.051,0.126,0.215,-0.185,-0.323,0.073,-0.047,-0.122,-0.070,-0.023,0.123,-0.147,0.085,-0.179,-0.041,-0.088,0.011,0.051,-0.124,-0.200,0.087,-0.143,0.001,-0.016,0.045,-0.134,0.064,-0.051,0.012,-0.024,0.030,0.020,-0.072,-0.385,-0.078,-0.157,0.107,0.047,0.050,0.024,0.134,-0.315,-0.021,-0.027,-0.068,-0.162,0.037,-0.007,0.037,-0.359,-0.127,-0.119,-0.165,-0.089,0.084,-0.212,0.174,-0.134,0.195,-0.361,0.046,0.028,0.011,-0.133,0.210,-0.043,0.250,-0.072,-0.062,0.083,-0.057,0.023,0.216,0.106,0.384,0.083,-0.293,0.115,-0.058,-0.115,0.208,0.122,0.250,0.197,-0.343,-0.014,-0.077,-0.285,0.119,0.088,0.272,-0.051,-0.337,-0.056,-0.097,-0.175,0.060,0.018,0.135,0.029,-0.199,0.071,-0.115,-0.133,0.131,-0.110,0.299,0.142,-0.066,0.054,-0.044,0.011,0.056,-0.075,0.216,0.024,0.109,0.086,-0.158,-0.008,-0.041,-0.079,0.107,0.021,0.002,0.079,-0.118,-0.034,-0.211,-0.193,0.074,0.062,0.093,-0.026,0.127,0.050,0.058,-0.145,0.069,0.055,-0.022,0.041,0.212,0.062,0.105,-0.289,-0.006,0.137,0.222,-0.188,-0.068,0.131,0.135,-0.025,0.103,0.029,-0.257,0.054,0.183,-0.238,0.173,-0.035,0.142,0.045,0.174,0.100,0.111,-0.096,0.344,0.044,0.030,-0.028,-0.161,0.364,0.024,-0.337,0.220,0.110,-0.282,0.204,0.148,0.152,0.060,-0.291,0.146,0.023,-0.073,0.187,-0.104,-0.020,0.193,-0.266,0.127,0.015,0.050,0.107,-0.234,-0.014,-0.080,-0.054,0.075,-0.064,0.001,-0.010,-0.031,0.017,-0.218,-0.091,0.043,-0.082,0.084,0.089,0.098,0.135,-0.259,0.022,-0.186,-0.133,0.144,0.177,0.058,0.025,-0.177,-0.042,-0.257,-0.100,-0.043,0.062,0.091,-0.046,-0.037,0.031,-0.316,0.051,-0.036,0.115,0.157,-0.192,0.291,0.118,-0.235,0.058,0.191,-0.071,0.399,0.229,0.094,-0.045,-0.395,0.095,0.146,0.095,0.101,-0.134,0.000,0.139,-0.319,-0.062,0.187,-0.139,-0.114,-0.054,-0.393,-0.190,-0.213,0.053,-0.222,0.053,-0.203,0.134,-0.311,-0.533,0.080,0.112,-0.131,0.166,-0.004,-0.031,0.029,-0.065,0.357,0.079,-0.164,0.120,-0.174,0.004,-0.116,-0.061,0.328,0.022,-0.028,0.304,-0.105,0.518,-0.244,-0.075,0.371,-0.044,0.113,0.372,-0.212,0.213,-0.014,-0.187,0.089,0.037,0.096,0.046,0.047,0.401,-0.137,0.009,-0.402,-0.141,-0.026,0.057,-0.013,0.399,0.135,-0.016,-0.557,-0.120,0.064,0.215,0.212,0.492,-0.003,-0.008,-0.253,-0.103,0.071,0.224,0.238,-0.211,0.209,0.132,-0.094,0.075,0.169,-0.142,0.315,0.024,-0.049,-0.158,0.323,-0.071,-0.031,0.058,0.061,0.181,0.253,0.174,0.086,0.249,-0.137,0.049,0.192,0.272,0.127,0.322,0.060,-0.158,-0.086,0.338,0.166,0.252,0.224,0.287,0.057,-0.159,-0.002,0.321,0.218,0.144,-0.006,0.230,-0.050,-0.320,0.379,0.168,0.183,0.082,0.046,0.029,-0.082,-0.450,0.472,0.005,0.172,0.229,0.043,0.142,0.009,-0.115,0.445,0.237,0.186,0.054,-0.267,0.022,-0.031,0.370,0.357,0.154,-0.096,0.055,0.117,-0.049,0.073,0.557,0.171,0.025,0.272,0.120,0.010,0.116,0.076,0.417,0.217,0.236,0.290,0.027,0.073,0.147,0.016,0.188,0.235,0.149,0.231,0.041,0.088,0.215,0.239,0.316,0.103,0.316,0.258,0.063,-0.130,-0.276,0.146,0.007,-0.046,0.010,-0.320,0.238,0.107,0.226,-0.054,0.123,-0.131,0.051,0.369,0.357,0.119,0.416,-0.146,-0.003,0.111,-0.047,0.047,0.064,0.126,0.312,-0.200,-0.332,0.322,-0.097,0.188,-0.199,-0.193,0.348,-0.189,-0.487,0.368,-0.077,0.195,-0.066,-0.173,0.111,-0.069,-0.620,0.449,-0.259,0.159,-0.209,0.151,-0.038,-0.107,-0.264,0.324,-0.062,-0.024,-0.102,-0.318,0.105,-0.124,-0.244,0.116,-0.118,-0.131,0.037,-0.114,-0.072,-0.106,-0.156,0.196,-0.100,0.154,0.011,0.068,-0.006,-0.198,-0.059,0.176,0.030,0.246,-0.006,0.126,-0.063,-0.021,-0.047,-0.010,-0.043,0.016,-0.253,0.096,0.067,0.080,0.042,-0.044,0.008,0.155,0.167,-0.151,-0.134,-0.358,0.079,-0.081,-0.127,-0.101,0.187,0.143,0.381,-0.372,0.088,0.001,-0.160,0.258,-0.010,-0.262,0.177,-0.146,-0.109,0.260,-0.104,0.073,-0.102,-0.056,0.212,-0.234,-0.765,0.336,-0.111,0.209,-0.102,-0.175,0.246,-0.195,-0.451,0.284,-0.084,-0.003,-0.053,-0.436,0.019,-0.285,-0.443,0.396,-0.043,0.020,-0.014,-0.208,-0.083,-0.107,-0.344,0.221,0.113,0.034,0.037,-0.284,-0.038,-0.124,-0.037,0.227,0.050,-0.114,0.013,-0.133,-0.151,-0.005,0.229,0.172,-0.104,0.010,-0.122,0.004,0.062,-0.068,0.196,0.026,-0.042,-0.242,0.043,-0.092,-0.084,-0.183,0.234,0.080,0.044,-0.162,-0.067,0.110,-0.135,-0.211,0.029,0.046,-0.058,-0.053,-0.067,-0.233,-0.265,-0.423,-0.073,-0.050,-0.049,0.172,-0.024,0.079,0.321,-0.271,0.080,0.071,-0.281,0.045,-0.033,-0.211,0.145,-0.050,-0.264,0.315,0.008,0.090,0.004,-0.089,0.174,-0.049,-0.873,0.266,-0.070,0.229,-0.117,-0.105,0.150,0.070,-0.768,0.248,-0.071,0.149,-0.024,-0.233,0.129,0.041,-0.285,0.133,-0.012,0.069,-0.145,-0.314,0.044,0.087,-0.017,0.276,0.028,0.074,0.039,-0.284,-0.173,0.117,-0.043,0.211,-0.071,0.006,-0.049,-0.202,-0.300,0.139,0.071,0.035,-0.112,-0.137,-0.140,-0.082,-0.056,0.052,0.173,0.068,0.004,-0.059,0.039,-0.153,-0.036,-0.029,0.224,0.083,-0.031,-0.235,-0.399,-0.056,-0.165,-0.220,0.161,-0.081,-0.542,-0.499,-0.138,-0.344,-0.104,-0.499,-0.291,-0.084,-0.306,0.006,-0.226,-0.042,0.250,0.009,-0.262,0.025,-0.174,-0.131,0.116,-0.088,0.154,0.047,-0.384,0.314,-0.112,0.105,0.007,0.028,0.122,0.119,-0.867,0.265,-0.052,0.166,-0.232,-0.063,-0.177,0.119,-0.789,0.168,0.137,0.047,-0.268,-0.031,0.072,0.346,-0.453,0.195,0.095,0.136,0.150,-0.106,-0.014,0.427,-0.355,0.078,0.088,0.079,0.217,-0.273,-0.317,0.325,0.104,0.081,0.185,0.009,0.158,-0.057,-0.428,0.190,0.296,-0.198,0.186,-0.322,0.035,0.060,-0.053,0.174,0.361,-0.143,-0.046,-0.396,-0.063,-0.013,-0.060,0.218,-0.019,-0.209,-0.055,-0.341,-0.083,-0.043,-0.067,-0.026,-0.557,-0.308,-0.100,-0.331,-0.105,-0.179,-0.223,-0.191,-0.199,0.032,-0.132,-0.385,-0.208,0.075,0.067,-0.096,-0.164,-0.132,-0.062,-0.131,0.125,-0.107,0.131,-0.147,-0.298,0.401,-0.004,0.157,-0.048,0.098,-0.036,0.012,-0.543,0.236,0.012,0.167,-0.202,0.164,0.135,0.077,-0.635,-0.093,0.053,0.023,-0.088,-0.143,0.000,0.270,-0.391,-0.254,0.045,0.069,-0.059,-0.211,-0.150,0.340,-0.035,-0.051,0.241,-0.048,0.157,-0.039,-0.216,0.211,0.185,-0.220,-0.050,-0.233,0.201,0.012,-0.192,0.308,0.257,-0.535,-0.144,-0.278,-0.240,0.026,-0.100,0.210,-0.083,-0.525,0.046,-0.306,-0.306,0.054,0.084,0.079,-0.118,-0.304,-0.037,-0.038,-0.019,0.124,0.073,0.022,-0.697,-0.042,-0.116,0.090,0.114,-0.282,-0.060,0.043,-0.105,-0.039,-0.499,-0.447,0.017,0.325,-0.114,-0.002,-0.211,-0.103,-0.124,0.071,0.177,0.002,0.111,-0.213,-0.386,0.355,-0.043,0.072,0.161,0.081,0.030,-0.273,-0.471,0.097,-0.009,0.200,-0.361,0.180,0.100,0.070,-0.740,-0.253,0.023,0.131,-0.057,0.113,0.225,0.233,-0.199,-0.188,0.234,0.147,0.095,-0.086,0.100,0.145,-0.054,0.053,0.214,0.125,0.227,0.072,-0.024,0.272,0.026,-0.304,0.229,0.077,0.152,0.014,-0.080,0.041,-0.181,-0.283,0.169,-0.283,-0.126,0.142,0.060,0.208,-0.222,-0.092,0.062,-0.154,-0.150,0.013,0.058,0.054,-0.271,-0.082,-0.108,0.043,-0.232,0.062,0.099,0.024,-0.214,-0.074,-0.185,0.304,-0.041,-0.278,-0.141,-0.251,0.072,-0.105,-0.270,-0.513,0.042,0.060,-0.346,-0.222,-0.105,-0.062,-0.092,0.109,0.168,-0.002,0.154,-0.233,-0.109,0.127,0.030,0.039,0.256,0.134,0.256,-0.284,-0.324,-0.174,0.111,0.098,0.019,0.224,-0.049,-0.342,-0.280,0.084,-0.075,0.142,-0.405,0.345,0.046,0.042,-0.352,-0.070,-0.042,0.018,-0.095,-0.033,0.064,0.104,-0.003,-0.186,0.023,0.109,0.152,0.118,0.149,0.179,0.121,-0.290,0.045,0.146,0.101,0.030,-0.029,0.125,0.080,0.023,-0.049,-0.032,0.083,-0.002,-0.007,-0.014,-0.165,0.069,-0.062,-0.093,0.050,-0.017,0.071,0.022,-0.095,-0.094,-0.023,0.172,-0.189,-0.003,0.011,0.140,-0.125,0.002,0.029,-0.001,-0.032,-0.088,-0.087,-0.163,0.123,-0.031,-0.068,-0.241,-0.180,-0.014,-0.162,-0.452,-0.079,-0.088,-0.330,0.119,0.018,0.016,0.276,-0.400,-0.008,-0.327,-0.074,0.156,0.061,0.116,0.191,-0.306,-0.015,-0.716,-0.062,0.206,-0.065,0.164,0.007,-0.156,-0.075,-0.104,-0.092,0.179,0.043,0.176,0.117,-0.024,-0.008,-0.353,0.024,0.247,-0.152,0.151,0.096,-0.014,0.008,-0.196,0.025,0.177,0.019,-0.070,0.167,-0.066,0.103,0.099,-0.048,-0.056,0.008,-0.000,0.097,0.062,0.146,0.140,-0.085,0.043,-0.092,-0.030,-0.035,-0.016,-0.079,0.092,-0.050,-0.132,-0.091,-0.115,-0.050,0.046,-0.100,-0.094,-0.056,0.149,-0.098,0.042,0.007,0.022,-0.110,0.007,-0.008,-0.107,0.060,-0.062,-0.155,-0.478,-0.058,-0.032,0.056,-0.143,-0.067,-0.038,0.095,-0.725,0.112,-0.203,0.072,-0.005,0.057,-0.079,-0.070,-0.673,0.034,-0.173,-0.066,0.065,0.072,0.126,0.206,-0.232,0.149,-0.444,-0.201,0.193,0.170,0.171,0.239,-0.135,0.040,-0.519,-0.247,0.282,-0.012,0.266,0.117,-0.001,0.183,-0.385,-0.100,-0.294,-0.027,0.122,0.082,0.060,0.188,-0.134,-0.006,-0.436,-0.021,-0.162,0.058,0.029,0.115,0.119,0.033,-0.158,0.023,-0.096,0.015,-0.048,0.004,0.020,-0.017,-0.038,-0.016,-0.113,-0.050,-0.009,-0.010,0.023,-0.019,-0.070,0.162,-0.277,-0.117,-0.235,-0.081,-0.019,0.024,-0.152,0.153,0.002,-0.057,-0.151,0.140,0.011,0.124,0.009,-0.106,-0.072,-0.031,-0.137,0.000,-0.063,-0.257,-0.163,-0.017,-0.169,-0.113,-0.633,0.055,-0.056,-0.282,-0.031,-0.038,-0.274,-0.048,-0.609,0.144,-0.232,0.014,0.000,-0.014,0.081,0.254,-0.014,0.107,0.004,0.127,0.070,-0.026,0.036,0.287,-0.312,0.088,-0.062,0.104,-0.109,-0.108,0.002,-0.040,0.116,0.222,-0.022,0.028,0.048,-0.020,-0.078,-0.178,0.244,0.081,0.019,-0.025,-0.293,-0.079,-0.167,-0.171,0.302,0.069,-0.027,0.004,-0.273,-0.111,-0.170,-0.115,-0.350,0.021,0.015,0.050,-0.037,-0.023,0.053,-0.002,-0.281,0.072,0.066,0.070,-0.031,-0.155,-0.084,0.034,0.025,-0.033,0.095,-0.051,-0.263,-0.198,-0.266,-0.388,-0.223,0.023,-0.087,-0.199,-0.337,-0.296,0.017,-0.076,0.133,-0.325,-0.004,-0.169,-0.130,-0.585,-0.102,-0.065,-0.144,-0.386,-0.103,-0.301,-0.548,0.005,-0.327,-0.221,-0.279,-0.014,-0.241,-0.195,-0.140,0.124,-0.281,-0.564,-0.307,0.045,0.011,-0.509,0.193,-0.045,-0.268,-0.154,-0.199,0.025,-0.025,-0.325,-0.060,-0.189,-0.275,-0.274,-0.405,-0.008,-0.080,-0.364,-0.146,-0.506,-0.311,-0.365,-0.625,-0.035,-0.164,0.088,-0.303,-0.854,-0.019,-0.126,-0.593,0.033,-0.031,0.263,-0.489,-0.415,-0.202,-0.237,-0.481,0.014,-0.013,0.173,-0.403,-0.355,0.297,0.020,-0.606,0.067,0.104,0.010,-0.348,-0.657,-0.098,0.002,-0.083,-0.051,-0.079,-0.172,-0.428,-0.364,-0.266,-0.513,-0.069,-0.429,-0.270,-0.071,-0.276,0.164,-0.144,0.008,0.044,-0.018,-0.049,0.026,-0.011,-0.164,0.078,-0.210,-0.066,-0.057,-0.004,-0.145,-0.120,-0.179,-0.427,0.110,0.095,-0.110,-0.013,-0.148,-0.100,-0.464,-0.191,0.162,-0.157,-0.118,0.101,-0.283,-0.279,-0.400,-0.256,-0.338,-0.103,-0.094,0.035,-0.424,-0.447,-0.910,-0.137,-0.164,-0.124,-0.067,0.074,-0.545,-0.195,-0.762,-0.657,-0.037,-0.128,-0.060,0.068,-0.482,-0.295,-0.891,-1.216,-0.177,-0.229,0.024,0.036,-0.502,-0.450,-0.982,-0.885,-0.232,-0.324,-0.048,0.082,-0.676,-0.790,-1.469,-0.085,-0.595,-0.441,-0.002,-0.029,-0.463,-0.595,0.158,-0.442,-0.175,-0.405,0.026,-0.028,-0.137,-0.073,0.095,-0.164,-0.802,-0.630,-0.027,0.032,-0.116,-0.216,-0.019,-0.024,0.243,-0.075,0.101,0.117,-0.298,0.017,-0.066,0.007,0.185,-0.007,0.214,-0.031,-0.294,-0.097,0.013,-0.093,0.063,0.116,0.313,0.022,-0.129,-0.195,-0.039,0.038,0.006,0.024,0.101,-0.230,-0.183,-0.101,-0.036,-0.128,-0.174,0.188,0.113,-0.372,0.207,-0.483,0.096,0.044,-0.153,-0.019,-0.037,-0.489,0.260,-0.050,0.190,0.019,-0.038,-0.108,-0.116,-0.438,0.159,-0.294,0.127,-0.098,-0.233,-0.028,-0.325,-0.215,0.010,-0.354,0.019,-0.167,-0.030,0.041,0.173,0.003,0.099,-0.201,-0.010,-0.094,-0.146,0.056,-0.215,0.024,-0.279,-0.137,-0.094,-0.239,0.030,-0.254,-0.024,-0.099,-0.406,0.004,-0.268,-0.417,-0.560,-0.574,-0.109,-0.049,-0.899,-0.484,0.091,0.481,0.372,0.135,0.020,0.088,-0.199,0.055,0.066,0.215,0.088,0.096,0.133,-0.076,-0.200,0.112,-0.177,0.218,0.410,0.122,0.238,-0.232,-0.099,0.178,0.049,0.275,0.249,0.107,0.375,-0.192,-0.119,0.102,-0.101,0.125,0.117,0.093,0.273,-0.254,-0.043,-0.097,-0.117,0.002,-0.106,-0.015,0.331,-0.211,0.123,0.167,-0.156,-0.065,-0.142,-0.066,0.187,-0.381,0.072,-0.139,-0.036,-0.165,-0.073,-0.053,0.206,0.036,0.040,-0.064,-0.193,-0.044,0.016,0.038,-0.262,0.169,0.147,-0.337,0.022,-0.109,-0.071,0.009,-0.198,0.062,0.158,-0.147,-0.120,-0.096,0.088,-0.045,-0.145,0.030,-0.095,-0.029,-0.057,-0.282,-0.094,-0.011,-0.380,-0.134,-0.262,-0.197,0.147,0.167,0.170,0.225,0.076,0.017,-0.045,-0.127,0.008,0.273,-0.215,0.040,0.069,-0.492,0.082,-0.097,-0.141,0.249,0.187,0.144,0.214,-0.490,-0.028,-0.030,-0.004,-0.017,-0.064,0.088,0.283,-0.327,-0.094,-0.014,-0.259,-0.039,0.058,0.247,0.313,-0.077,-0.126,-0.248,-0.329,-0.353,-0.072,0.111,0.401,0.172,-0.005,-0.224,-0.354,0.025,-0.006,0.065,0.338,-0.295,0.005,-0.093,-0.288,-0.056,-0.014,0.121,0.075,0.116,0.166,0.067,-0.175,0.002,0.100,0.198,-0.061,0.255,-0.011,0.021,-0.141,0.072,0.049,0.042,-0.262,0.268,0.017,-0.232,-0.076,-0.009,0.104,-0.055,-0.287,0.176,0.041,0.296,0.225,-0.363,-0.073,0.100,-0.221,0.038,0.065,-0.282,0.061,0.336,-0.106,0.222,0.179,-0.137,0.110,-0.050,-0.038,0.252,-0.294,-0.127,0.190,-0.371,-0.019,0.121,-0.273,0.022,-0.134,0.137,0.145,-0.239,-0.246,-0.140,-0.307,-0.343,-0.237,0.161,0.200,-0.060,0.054,-0.325,-0.306,-0.138,-0.154,0.044,0.250,0.244,-0.051,-0.426,-0.335,-0.319,-0.099,0.042,0.252,0.349,-0.171,-0.293,-0.300,-0.003,0.154,-0.134,0.050,-0.034,0.009,-0.032,-0.153,-0.004,0.127,-0.158,-0.027,0.159,0.008,0.081,0.141,-0.054,0.080,0.064,-0.072,0.180,0.126,0.077,0.108,0.010,0.087,0.053,-0.005,0.157,0.112,-0.139,0.122,0.052,0.075,-0.015,-0.207,0.189,0.203,0.130,0.021,-0.011,-0.114,0.022,-0.014,0.095,0.071,-0.371,0.253,0.122,-0.057,0.093,0.203,0.069,0.401,-0.072,-0.028,0.017,-0.434,-0.006,-0.044,0.015,-0.152,-0.125,-0.171,-0.127,-0.338,-0.229,0.212,0.090,0.016,-0.164,0.001,-0.312,-0.226,-0.009,0.278,0.193,0.180,-0.220,-0.316,0.015,-0.203,-0.045,0.111,0.409,-0.056,-0.191,-0.371,-0.058,-0.146,-0.255,-0.010,0.464,-0.298,-0.305,-0.376,-0.005,0.174,-0.235,-0.040,0.089,-0.233,0.281,0.188,0.010,0.029,-0.203,-0.022,-0.138,0.161,0.322,0.208,0.086,0.020,0.015,-0.014,-0.084,-0.035,0.212,-0.178,-0.084,0.081,-0.064,-0.029,0.022,-0.215,-0.128,-0.173,-0.094,0.003,-0.033,0.082,0.008,-0.162,-0.069,-0.070,-0.007,-0.162,-0.067,0.144,0.001,0.043,-0.418,0.115,-0.237,0.051,-0.194,0.041,0.267,0.166,0.272,-0.271,-0.215,-0.138,-0.390,-0.051,0.591,0.037,-0.034,-0.229,-0.161,-0.195,0.052,-0.191,0.534,0.015,-0.178,-0.122,-0.126,-0.110,0.146,0.055,0.619,0.131,-0.292,-0.206,-0.253,-0.268,-0.090,-0.019,0.949,-0.020,-0.069,-0.287,-0.310,-0.407,-0.191,-0.279,0.471,-0.129,-0.381,-0.102,-0.112,-0.077,-0.140,-0.277,0.082,-0.150,0.249,-0.049,-0.053,-0.040,-0.121,0.055,-0.161,0.022,0.204,-0.059,-0.049,0.076,0.033,0.099,-0.030,0.095,-0.066,0.025,-0.175,0.016,0.050,0.026,0.090,0.011,-0.019,-0.021,-0.156,0.114,0.040,-0.121,-0.009,0.020,0.029,0.007,-0.320,-0.089,-0.078,-0.105,0.049,0.079,-0.086,-0.176,-0.334,0.022,-0.421,-0.063,0.235,-0.007,0.137,-0.119,-0.239,-0.171,-0.338,-0.124,0.549,-0.037,0.046,-0.058,-0.046,-0.162,-0.023,-0.125,0.620,0.040,-0.153,0.102,-0.155,0.087,0.046,0.058,0.536,0.280,-0.140,0.084,-0.142,-0.073,0.005,0.056,0.670,-0.090,-0.067,-0.190,-0.181,-0.148,0.003,-0.059,-0.064,-0.270,-0.269,-0.161,0.003,-0.101,0.041,-0.089,0.077,-0.088,0.015,-0.187,-0.085,-0.027,0.165,-0.158,-0.040,0.046,0.172,-0.009,-0.249,0.034,0.124,0.144,0.159,-0.058,0.219,-0.087,-0.145,-0.101,0.173,0.130,0.265,0.044,0.250,-0.192,-0.098,0.068,0.007,-0.003,0.191,-0.111,0.044,-0.195,-0.022,0.062,-0.071,-0.274,0.024,0.061,-0.081,-0.135,-0.493,-0.109,-0.141,-0.156,0.068,-0.005,-0.409,-0.285,-0.229,-0.083,-0.024,-0.282,0.449,-0.183,-0.173,0.162,-0.123,-0.061,-0.142,-0.040,0.356,0.089,-0.225,0.257,-0.027,0.088,-0.082,-0.016,0.393,0.174,0.128,-0.100,-0.280,0.156,-0.228,-0.053,0.105,-0.161,0.041,-0.069,-0.114,0.075,-0.009,-0.101,0.042,-0.077,-0.088,-0.146,-0.153,0.015,-0.492,-0.061,-0.013,-0.017,-0.007,-0.363,-0.177,-0.043,0.006,-0.103,-0.013,-0.008,-0.025,-0.138,-0.088,-0.050,0.180,-0.013,-0.013,0.167,-0.164,0.138,-0.120,-0.003,0.237,0.034,0.166,0.355,-0.041,0.091,-0.082,-0.087,0.059,-0.116,0.156,0.202,-0.253,-0.129,0.075,-0.062,-0.028,-0.050,0.046,0.164,0.032,-0.081,-0.397,-0.189,-0.059,-0.145,-0.027,-0.170,-0.186,-0.002,-0.357,-0.229,-0.029,-0.197,0.158,0.233,-0.097,0.050,-0.279,-0.244,-0.030,-0.133,0.140,0.061,0.043,0.006,-0.351,0.091,-0.346,-0.137,0.131,-0.223,-0.104,0.059,-0.121,0.170,-0.375,-0.221,0.335,0.061,0.068,-0.074,0.109,0.325,-0.382,-0.265,0.123,-0.017,0.033,-0.024,0.009,0.144,-0.106,-0.166,-0.127,-0.247,0.008,-0.146,-0.094,-0.002,-0.118,-0.207,-0.188,-0.102,0.010,-0.017,0.009,0.031,-0.236,-0.215,-0.026,0.080,-0.029,-0.326,0.081,0.129,-0.164,-0.119,0.096,0.018,-0.157,-0.398,-0.111,-0.041,-0.245,-0.025,0.149,-0.316,-0.634,-0.201,-0.056,-0.285,-0.218,0.008,0.048,-0.369,-0.149,-0.060,0.109,-0.009,0.173,-0.334,-0.005,-0.201,-0.626,-0.075,-0.484,-0.308,0.114,-0.248,0.054,-0.116,-0.078,0.242,-0.253,0.051,-0.090,-0.302,0.129,0.056,0.009,0.058,-0.448,0.107,-0.303,-0.355,0.482,0.053,-0.001,-0.057,-0.017,0.132,-0.538,-0.260,0.163,-0.058,0.186,-0.043,0.159,0.248,-0.490,-0.175,0.226,-0.229,0.050,-0.006,0.053,0.187,-0.383,0.016,0.021,-0.233,0.082,-0.280,0.075,0.067,-0.344,-0.157,-0.003,-0.186,0.096,-0.281,0.056,0.043,-0.069,-0.276,-0.038,-0.079,0.060,-0.816,0.075,0.048,-0.329,-0.124,-0.056,-0.247,-0.020,-0.221,0.053,-0.037,-0.056,-0.019,0.034,-0.154,-0.132,-0.537,-0.338,-0.250,-0.019,-0.268,0.002,-0.308,-0.458,-0.031,-0.064,-0.090,-0.365,-0.055,-0.025,0.123,-0.398,0.144,-0.214,0.013,-0.567,0.050,0.094,0.036,0.127,0.014,0.114,0.271,0.018,-0.145,0.236,0.001,0.042,-0.049,0.097,0.194,-0.218,-0.213,0.295,-0.131,0.054,0.060,-0.107,0.255,-0.185,-0.238,0.258,-0.011,0.127,0.111,0.051,0.285,0.023,-0.120,0.098,-0.227,-0.046,0.099,0.107,0.269,0.074,-0.093,0.074,-0.117,0.303,-0.241,0.136,0.176,-0.015,-0.050,0.097,-0.280,0.191,-0.341,0.239,0.016,-0.183,-0.043,-0.013,-0.199,0.072,-0.361,0.029,0.095,-0.038,-0.248,-0.029,-0.267,0.046,-0.128,0.080,-0.074,0.077,-0.182,-0.095,-0.311,0.263,-0.065,-0.264,-0.208,-0.137,-0.433,-0.070,-0.181,-0.094,-0.380,-0.142,-0.229,-0.498,0.101,-0.100,-0.062,-0.198,0.181,-0.222,-0.084,0.075,-0.016,-0.105,-0.134,-0.051,-0.389,0.111,0.124,0.146,-0.102,-0.185,-0.302,0.066,-0.341,0.007,-0.218,0.041,-0.310,-0.096,-0.033,-0.256,0.091,-0.184,0.153,0.073,-0.154,-0.190,0.252,0.109,0.065,0.016,0.089,0.098,-0.053,-0.286,0.107,0.002,0.108,-0.086,0.043,0.057,-0.421,-0.330,-0.125,0.172,0.101,0.193,0.158,0.092,0.149,-0.151,-0.033,0.179,0.323,-0.294,-0.028,-0.080,-0.152,0.020,0.040,-0.201,-0.160,-0.010,-0.132,-0.092,0.059,-0.094,-0.017,-0.098,-0.052,-0.061,-0.063,-0.102,0.130,-0.105,-0.189,-0.379,-0.271,0.001,-0.069,0.044,0.104,-0.126,-0.254,-0.348,-0.333,-0.328,-0.207,-0.134,-0.372,-0.158,-0.269,-0.077,-0.352,-0.357,-0.575,0.018,-0.119,-0.156,-0.425,-0.118,0.082,-0.144,-0.150,0.099,-0.413,-0.155,0.192,0.234,0.069,-0.242,0.006,0.056,-0.485,-0.068,0.176,-0.015,0.061,0.010,-0.102,0.157,0.094,-0.134,0.028,-0.121,-0.065,-0.051,-0.030,0.208,0.090,-0.083,0.120,-0.054,0.197,-0.183,0.049,0.194,-0.307,0.074,0.076,-0.128,0.082,0.077,0.053,0.165,-0.035,-0.015,0.117,0.086,0.109,0.007,0.139,0.071,0.156,-0.215,0.032,-0.224,0.025,-0.038,0.003,-0.034,0.347,-0.074,0.024,0.259,0.060,0.042,0.009,0.035,0.095,0.014,-0.016,0.002,0.012,0.018,-0.074,-0.186,-0.070,-0.087,-0.052,-0.053,-0.329,0.152,-0.152,0.027,-0.450,-0.061,-0.207,-0.314,0.033,-0.280,-0.427,-0.108,-0.069,-0.221,0.119,0.027,0.065,-0.101,-0.119,0.125,-0.005,-0.021,0.178,-0.008,0.106,-0.051,0.167,0.107,-0.265,-0.141,0.001,0.035,0.190,0.036,-0.114,0.064,-0.099,-0.170,0.171,0.034,-0.029,0.119,-0.092,0.078,-0.176,-0.149,0.179,-0.082,0.193,0.005,0.038,0.021,-0.103,-0.204,0.212,-0.003,0.157,0.161,0.115,0.018,-0.003,-0.203,0.010,-0.139,-0.004,0.046,0.097,0.080,0.067,-0.150,0.054,-0.028,0.075,-0.178,0.017,0.104,0.137,-0.129,-0.174,0.103,0.138,-0.123,0.072,0.135,0.193,-0.043,0.172,0.087,0.158,-0.087,0.092,-0.042,0.047,-0.047,-0.024,0.010,0.003,0.005,-0.408,-0.047,-0.099,-0.091,-0.053,0.079,0.051,-0.026,0.142,0.031,-0.086,-0.168,0.355,0.050,0.114,-0.024,-0.011,0.066,-0.153,-0.148,-0.041,0.187,0.134,0.059,0.078,0.207,-0.068,-0.152,0.082,0.170,0.218,0.120,-0.041,0.083,-0.012,-0.297,0.213,0.064,0.071,0.408,-0.139,0.282,-0.172,-0.303,0.014,-0.053,0.081,0.207,0.043,0.083,-0.122,-0.330,0.070,0.031,0.146,0.022,-0.069,-0.124,0.028,-0.162,0.127,0.054,0.104,-0.118,-0.020,0.070,0.140,-0.082,0.072,-0.058,0.355,-0.101,0.044,0.115,0.046,-0.038,-0.123,-0.159,0.022,0.116,0.091,0.167,-0.039,-0.061,-0.030,0.009,-0.006,-0.151,0.119,0.127,-0.100,-0.144,-0.028,-0.042,-0.102,-0.198,-0.256,0.338,-0.090,-0.133,0.108,-0.109,0.145,0.220,0.188,0.008,-0.009,-0.223,-0.112,-0.000,-0.059,0.070,0.131,0.127,-0.104,-0.368,-0.234,0.045,0.136,0.096,0.135,0.003,-0.059,-0.395,-0.091,0.198,0.256,0.214,0.193,0.172,-0.065,-0.475,-0.201,0.242,0.097,0.455,0.126,0.134,-0.062,-0.503,-0.230,0.123,0.008,0.342,0.077,0.306,-0.058,-0.255,-0.022,0.034,0.040,0.077,0.038,0.063,-0.269,0.131,0.193,0.117,0.232,-0.213,0.001,-0.112,-0.153,0.081,0.002,0.090,0.311,-0.184,-0.034,0.097,0.093,0.097,0.043,0.166,0.150,-0.046,0.087,0.062,0.016,0.028,-0.085,-0.107,0.079,0.034,0.145,0.038,-0.148,0.028,0.049,-0.087,0.248,-0.232,-0.286,-0.074,0.152,-0.133,0.152,-0.012,0.001,0.181,0.143,0.021,0.084,-0.508,-0.030,0.056,0.085,0.165,0.053,-0.111,0.009,-0.623,-0.183,0.078,-0.203,0.249,0.256,-0.023,-0.009,-0.592,-0.261,0.172,0.017,0.238,0.221,-0.015,-0.100,-0.398,-0.322,-0.088,0.208,0.437,0.132,0.050,-0.191,-0.531,-0.137,-0.016,-0.322,0.215,-0.047,0.136,-0.214,-0.341,0.175,0.060,0.043,-0.130,-0.097,-0.033,-0.103,0.078,0.210,0.155,0.263,-0.084,-0.008,0.040,-0.051,0.222,0.109,0.154,0.346,-0.006,-0.014,0.078,0.053,0.108,0.097,0.059,0.400,0.056,-0.072,-0.096,-0.000,0.104,0.188,0.271,-0.291,0.064,0.041,-0.301,0.058,0.092,-0.051,-0.053,0.135,-0.234,-0.214,-0.099,0.192,-0.162,0.036,-0.006,-0.015,-0.027,-0.718,0.025,0.045,-0.198,-0.044,0.002,0.006,0.117,0.027,0.061,0.009,-0.462,0.001,-0.056,0.217,0.126,0.040,0.135,-0.105,-0.690,-0.121,-0.028,0.408,0.246,0.232,0.148,-0.179,-0.313,0.103,-0.004,0.040,0.233,0.116,0.151,-0.126,-0.096,-0.068,0.143,-0.013,0.207,-0.014,0.257,-0.157,-0.151,0.137,-0.044,0.136,0.000,-0.072,0.100,0.063,-0.195,0.147,-0.126,0.292,-0.008,-0.156,-0.019,0.144,-0.213,0.005,-0.105,0.149,0.081,-0.041,-0.072,0.164,-0.062,0.189,-0.066,0.067,0.061,-0.078,-0.117,0.116,-0.029,0.044,-0.158,-0.329,0.129,-0.008,-0.077,-0.081,-0.149,-0.131,-0.252,-0.082,-0.313,-0.059,-0.045,0.053,-0.035,-0.142,-0.089,0.089,-0.588,-0.224,-0.015,-0.035,-0.200,0.190,0.059,0.166,-0.153,0.181,-0.060,-0.204,-0.081,0.150,0.025,0.110,-0.039,-0.118,-0.193,-0.304,0.188,0.079,0.067,0.211,0.083,0.070,0.261,-0.284,0.265,0.240,0.073,0.110,0.139,-0.094,-0.021,-0.231,-0.074,0.222,0.372,0.078,0.209,0.057,0.113,-0.048,-0.310,0.060,-0.001,0.424,0.061,0.058,-0.010,0.139,-0.340,0.209,-0.175,0.201,0.143,0.047,0.243,0.103,-0.375,0.011,-0.117,-0.241,0.068,0.010,0.058,0.201,-0.341,-0.367,-0.196,-0.155,-0.032,-0.101,0.059,-0.100,-0.263,-0.144,-0.020,0.063,0.219,0.002,-0.053,-0.281,-0.031,-0.076,0.045,-0.049,-0.465,-0.242,-0.081,-0.063,-0.065,-0.226,-0.582,0.072,-0.076,-0.154,-0.021,-0.077,-0.165,0.221,-0.016,-0.066,-0.071,0.201,-0.203,-0.288,-0.032,0.200,0.022,0.056,-0.021,0.203,-0.169,-0.451,0.173,0.256,0.059,0.166,0.177,0.155,-0.300,-0.357,0.062,0.292,0.196,0.042,0.303,0.174,-0.189,-0.253,-0.298,0.029,0.257,-0.034,0.175,0.031,0.052,-0.054,-0.375,-0.028,-0.100,0.124,0.092,0.065,0.159,-0.102,-0.138,0.058,-0.004,0.092,-0.098,-0.003,0.051,-0.215,-0.003,-0.023,-0.031,0.012,0.049,0.030,0.100,-0.150,-0.211,-0.058,0.130,0.121,-0.052,-0.048,-0.084,-0.230,-0.159,0.065,0.194,-0.232,0.069,-0.028,-0.098,-0.141,-0.044,-0.082,0.269,-0.070,-0.405,-0.075,0.052,-0.197,-0.036,-0.263,-0.618,-0.117,0.035,-0.057,0.016,-0.096,-0.226,0.115,0.054,0.165,-0.027,0.207,-0.078,-0.175,-0.715,0.058,0.207,0.006,0.001,0.022,-0.253,-0.183,-0.619,-0.009,0.207,-0.069,0.171,0.167,-0.195,-0.028,-0.603,0.016,0.222,0.049,0.140,0.085,0.052,0.065,-0.422,0.099,0.251,-0.119,0.072,-0.005,-0.073,-0.060,-0.009,0.041,-0.064,-0.017,-0.081,0.074,-0.023,-0.029,-0.125,0.073,0.114,-0.019,-0.030,-0.053,-0.061,-0.147,-0.045,-0.053,0.300,0.040,0.043,-0.000,-0.265,-0.210,-0.051,-0.069,0.083,-0.007,0.106,-0.066,-0.294,-0.181,-0.120,-0.185,0.273,-0.453,-0.316,0.005,-0.221,-0.021,-0.151,-0.370,-0.372,-0.079,-0.272,-0.008,-0.502,0.051,-0.168,-0.089,-0.377,0.213,0.112,0.149,-0.560,0.139,-0.212,0.192,0.070,-0.073,0.118,0.193,-0.411,0.120,-0.123,-0.134,0.026,-0.018,0.199,0.214,-0.179,0.062,-0.263,-0.254,0.059,0.030,0.017,0.227,-0.122,0.084,-0.219,-0.077,0.047,-0.003,-0.036,-0.039,0.004,0.063,-0.132,-0.066,0.102,0.106,-0.159,-0.116,-0.136,0.005,-0.142,0.096,0.026,0.171,-0.151,-0.070,-0.392,-0.040,-0.062,-0.102,0.011,0.224,-0.106,-0.056,0.001,0.005,-0.112,-0.017,0.105,0.107,-0.038,0.063,-0.258,-0.090,0.022,0.043,0.071,-0.008,-0.010,-0.020,-0.123,-0.123,-0.041,0.000,0.191,-0.079,-0.324,-0.041,0.075,-0.203,-0.035,-0.072,-0.624,-0.024,-0.144,-0.039,-0.542,0.015,-0.051,0.069,-0.175,-0.038,0.113,0.043,-1.150,0.038,-0.066,-0.138,0.103,-0.106,0.085,-0.163,-0.153,0.107,-0.040,-0.251,-0.031,-0.256,-0.045,-0.260,-0.348,0.031,-0.162,-0.130,-0.076,0.054,-0.216,-0.196,-0.332,0.056,-0.043,0.084,-0.170,0.037,-0.191,-0.253,-0.251,-0.039,-0.015,0.073,-0.016,0.125,-0.308,-0.251,-0.358,0.048,0.038,0.152,-0.063,0.305,-0.036,-0.002,-0.375,0.002,-0.084,0.110,0.092,-0.032,-0.068,-0.136,-0.542,-0.009,0.014,0.207,-0.152,-0.284,-0.116,-0.095,-0.395,0.124,-0.040,-0.109,-0.143,-0.531,0.109,0.044,-0.393,0.117,-0.014,-0.296,-0.074,-0.662,-0.230,0.029,-0.031,0.029,-0.089,-0.214,-0.477,-0.002,-0.100,-0.000,-0.350,0.193,0.012,0.028,-0.053,-0.381,-0.146,-0.098,-0.193,0.192,0.005,-0.021,-0.101,-0.493,-0.340,-0.452,-0.112,0.190,-0.067,-0.127,-0.134,0.138,-0.015,0.011,0.126,0.063,0.017,-0.232,-0.124,-0.361,0.137,-0.092,0.161,-0.149,0.001,-0.402,-0.030,-0.747,0.148,-0.058,0.080,-0.070,-0.142,-0.448,-0.053,-0.356,0.201,-0.086,0.023,-0.260,-0.082,-0.657,-0.129,-0.350,-0.137,-0.066,0.004,-0.165,-0.062,-0.452,-0.139,-0.178,-0.096,0.196,0.015,0.069,-0.079,-0.164,-0.243,-0.014,-0.318,-0.111,-0.210,0.121,-0.007,-0.686,-0.173,-0.427,-0.281,-0.420,-0.844,0.121,-0.061,-0.389,-0.235,-0.379,-0.168,-0.857,-0.988,0.132,-0.003,-0.192,-0.187,-0.403,-0.226,-0.140,-0.442,0.155,-0.006,-0.033,-0.123,-0.210,-0.167,-0.489,0.062,0.083,0.023,-0.196,-0.221,-0.151,0.088,-0.588,0.154,-0.377,-0.042,-0.355,-0.125,-0.684,-0.054,-0.562,0.112,-1.037,-0.036,-0.012,-0.499,0.193,-0.304,-0.503,0.059,-1.427,0.100,0.320,-0.165,0.261,-0.053,-0.006,0.199,-1.191,0.103,0.210,0.059,0.006,-0.099,-0.189,0.153,-0.219,-0.285,0.113,-0.216,-0.173,-0.045,-0.145,0.164,0.137,-0.315,0.343,-0.087,-0.040,-0.010,-0.085,0.127,-0.480,0.014,0.068,-0.094,-0.066,0.193,0.001,0.057,-0.259,-0.331,-0.057,0.029,0.008,-0.330,-0.074,-0.049,-0.120,-0.075,-0.093,-0.271,-0.805,-0.126,-0.153,-0.119,-0.077,-0.140,-0.937,-0.387,-0.577,-0.318,-0.304,0.107,-0.448,-0.004,-0.231,-0.271,-0.540,-0.325,-0.734,0.011,-0.979,-0.115,-0.076,-0.488,0.184,-0.284,-1.022,0.114,-0.561,-0.108,0.294,0.019,-0.003,-0.477,-0.667,0.049,-0.441,-0.114,0.452,0.004,0.176,-0.477,-0.636,0.240,-0.123,-0.070,0.307,-0.051,0.189,-0.055,-0.354,0.229,0.065,-0.125,0.204,-0.172,0.327,0.042,-0.084,0.145,0.092,-0.205,0.020,-0.420,0.354,0.027,-0.006,0.171,0.190,-0.190,0.176,-0.330,0.239,0.223,0.006,0.070,0.307,-0.371,-0.176,-0.302,-0.009,0.126,-0.016,0.111,0.210,-0.280,-0.064,-0.198,-0.068,0.127,-0.052,0.039,-0.031,-0.263,-0.109,-0.122,-0.135,0.124,-0.196,-0.154,-0.472,-0.153,0.017,-0.426,-0.314,-0.446,-0.504,-0.071,-0.263,-0.067,0.002,-0.171,0.027,-0.574,-0.311,0.082,-0.193,-0.018,0.216,-0.103,0.192,-0.033,-0.112,-0.087,-0.163,0.029,0.255,0.026,0.175,0.056,0.022,0.079,-0.117,0.129,0.160,-0.056,0.260,0.192,0.046,0.146,-0.100,-0.102,0.329,0.104,0.298,0.278,-0.027,0.153,-0.132,-0.233,0.305,0.156,0.348,0.212,-0.073,0.060,-0.018,-0.214,0.160,0.091,0.203,0.002,-0.042,0.102,0.155,-0.050,0.004,-0.144,0.029,0.098,-0.042,-0.120,0.040,-0.272,0.016,-0.046,-0.045,-0.021,-0.051,-0.035,0.029,-0.286,-0.074,-0.072,-0.195,-0.059,-0.054,-0.092,0.083,-0.176,-0.201,-0.210,-0.483,0.085,-0.107,-0.125,-0.121,-0.220,-0.312,-0.446,-0.023,-0.311,-0.249,-0.096,-0.058,-0.038,-0.059,0.086,-0.062,-0.000,0.145,0.082,-0.152,0.123,0.010,0.024,0.160,0.201,0.168,-0.034,-0.199,-0.108,0.147,0.126,0.193,0.315,0.131,-0.049,-0.072,-0.170,0.030,0.123,0.076,0.234,0.220,-0.049,-0.043,-0.162,0.085,0.201,0.131,0.256,0.082,-0.237,-0.019,-0.109,0.227,0.309,0.384,0.051,-0.154,-0.005,-0.147,-0.048,0.222,0.151,0.312,-0.080,-0.112,0.019,0.025,-0.117,0.061,-0.219,0.121,0.044,-0.005,-0.108,0.236,-0.052,-0.045,-0.050,0.006,-0.005,-0.044,0.008,-0.074,-0.224,-0.101,-0.233,-0.302,0.102,-0.032,0.215,-0.238,-0.195,0.012,-0.186,-0.090,-0.077,0.053,0.216,-0.254,-0.069,-0.103,-0.222,-0.106,-0.051,0.001,-0.127,-0.126,-0.116,0.051,-0.030,0.142,0.098,0.442,-0.068,0.053,-0.121,0.141,0.208,-0.021,0.156,0.250,-0.023,-0.053,-0.270,-0.003,0.083,0.021,0.275,0.259,-0.159,0.036,-0.323,-0.276,0.081,-0.042,0.069,0.123,-0.094,0.054,-0.215,-0.178,0.196,-0.068,-0.084,-0.006,-0.190,0.089,-0.143,0.065,0.072,0.260,-0.001,-0.276,-0.147,0.297,-0.292,0.113,-0.145,0.116,0.073,0.058,-0.135,-0.015,0.323,0.018,-0.177,-0.170,0.040,-0.051,-0.123,-0.177,-0.047,0.088,-0.058,-0.056,0.022,0.073,-0.082,-0.182,0.009,0.089,-0.195,-0.440,0.030,0.147,-0.181,-0.318,0.084,-0.123,-0.099,-0.301,-0.208,0.062,-0.138,-0.067,0.097,-0.370,-0.271,0.078,0.122,-0.107,-0.155,-0.098,-0.133,-0.010,0.038,0.069,0.288,0.239,-0.136,-0.086,-0.486,-0.104,0.016,-0.107,0.035,0.025,-0.175,-0.075,-0.396,-0.124,0.188,-0.251,0.017,0.280,-0.149,0.085,-0.263,-0.441,0.053,-0.180,0.003,0.248,-0.013,0.098,-0.477,-0.030,-0.096,0.103,-0.160,-0.190,-0.222,0.223,0.081,0.139,-0.246,0.196,-0.105,-0.091,-0.283,0.368,-0.010,0.141,-0.112,-0.142,0.103,0.054,-0.258,0.020,0.240,0.027,-0.273,-0.219,0.034,-0.017,-0.107,-0.311,0.204,0.130,-0.082,-0.124,-0.012,0.123,-0.200,-0.393,0.103,0.005,0.031,-0.094,0.129,0.035,-0.250,-0.198,0.143,-0.023,0.160,-0.410,-0.076,-0.064,-0.435,-0.325,-0.061,-0.366,-0.724,-0.117,-0.134,-0.410,-0.510,-0.007,-0.089,-0.077,-0.104,-0.025,-0.075,0.134,-0.267,0.018,-0.446,0.037,-0.001,-0.157,0.096,0.056,-0.093,0.129,-0.434,-0.301,0.233,-0.199,0.121,0.071,0.168,0.144,-0.570,-0.199,-0.046,0.144,-0.025,-0.018,0.100,0.235,-0.501,-0.065,-0.137,0.263,-0.479,-0.476,-0.251,0.131,-0.069,-0.005,-0.465,0.376,-0.175,-0.103,-0.406,0.197,-0.034,-0.055,-0.094,-0.073,-0.005,0.066,-0.379,0.081,0.105,-0.071,0.062,0.003,0.026,0.045,-0.461,-0.195,-0.045,0.123,-0.035,-0.134,0.018,0.173,-0.023,-0.367,0.064,0.193,0.041,-0.387,-0.067,0.104,-0.066,-0.332,0.014,-0.114,0.017,-0.090,-0.266,-0.251,-0.185,-0.474,-0.103,-0.449,-0.177,-0.072,-0.614,-0.370,-0.246,-0.130,-0.094,-0.104,-0.059,0.030,0.075,0.027,-0.272,0.044,-0.301,0.040,0.156,-0.190,0.018,-0.008,0.184,0.053,-0.241,0.218,0.124,-0.150,-0.035,-0.198,0.127,0.166,-0.516,-0.263,-0.302,0.093,-0.185,-0.387,0.211,0.122,-0.443,0.078,-0.267,0.218,-0.497,-0.193,-0.329,0.061,-0.178,0.100,-0.216,0.027,-0.232,-0.081,-0.150,-0.107,0.116,-0.173,0.041,-0.053,0.141,-0.062,-0.457,-0.134,0.140,-0.002,0.107,-0.149,0.092,0.055,-0.413,-0.249,-0.036,0.025,-0.096,-0.017,-0.005,0.210,-0.115,-0.378,-0.002,-0.098,-0.206,-0.070,-0.087,0.169,0.061,-0.225,0.032,-0.088,0.060,0.053,-0.049,-0.051,-0.003,-0.358,-0.043,-0.022,0.009,-0.114,-0.257,-0.388,-0.121,-0.050,0.003,-0.231,-0.278,-0.069,0.202,-0.231,-0.038,-0.049,-0.060,-0.058,0.172,-0.142,0.010,0.043,0.329,0.133,-0.055,-0.304,-0.150,-0.117,-0.249,-0.335,0.364,0.139,-0.048,0.107,-0.162,0.017,-0.217,-0.081,0.383,0.152,0.251,0.042,-0.251,0.076,-0.113,0.090,0.088,-0.042,0.194,-0.107,-0.165,-0.156,-0.095,0.040,-0.110,-0.269,0.027,-0.180,0.167,-0.022,-0.029,-0.016,-0.216,-0.265,0.031,0.059,0.093,-0.019,-0.030,0.001,-0.164,-0.136,0.074,0.106,0.029,-0.131,0.170,0.015,-0.023,0.086,-0.026,0.087,0.003,-0.136,0.106,0.039,0.211,0.059,-0.017,-0.109,0.010,-0.262,0.239,0.003,0.274,0.077,-0.049,-0.168,0.285,0.180,-0.145,-0.035,0.016,0.116,0.024,-0.065,-0.057,0.072,0.019,-0.177,0.107,-0.025,0.033,-0.032,0.148,0.039,-0.385,-0.398,0.223,-0.035,-0.090,0.064,-0.145,-0.077,-0.184,-0.239,0.192,-0.155,0.043,0.113,-0.048,-0.188,-0.300,-0.064,0.223,-0.252,0.064,0.085,-0.126,0.119,-0.231,0.136,-0.084,-0.326,0.081,-0.013,0.053,-0.222,0.058,0.106,-0.057,-0.087,-0.079,0.035,0.108,0.078,0.054,0.103,-0.020,-0.058,-0.086,0.095,0.099,-0.164,0.201,0.088,-0.071,-0.039,-0.123,-0.007,-0.037,-0.058,0.214,0.174,0.132,-0.008,-0.112,0.040,0.042,-0.113,-0.004,0.079,0.226,-0.109,-0.131,-0.220,-0.173,0.164,0.216,0.082,0.256,0.238,0.078,0.384,0.333,0.022,0.022,-0.076,0.395,0.002,-0.004,0.028,-0.169,-0.068,0.059,-0.302,0.547,-0.037,-0.045,-0.062,0.001,0.000,0.059,0.265,0.160,-0.134,0.172,-0.041,0.037,-0.187,0.253,0.356,0.089,-0.069,0.127,-0.010,-0.159,-0.059,0.255,0.224,-0.473,-0.180,0.034,0.032,0.057,-0.101,0.350,0.360,-0.073,-0.204,-0.002,-0.217,0.141,-0.349,0.330,0.453,0.257,-0.015,-0.055,-0.246,0.120,-0.215,0.125,0.324,0.310,-0.013,-0.015,-0.206,0.206,-0.274,0.107,0.103,0.194,-0.038,-0.023,-0.283,0.009,-0.401,0.295,0.264,0.356,-0.008,-0.053,-0.277,0.014,0.014,0.130,0.255,0.397,0.097,0.069,0.066,-0.036,-0.122,0.107,0.073,0.114,0.329,0.030,0.217,0.031 
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
  
