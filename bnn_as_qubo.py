import qubovert as qv
from qubovert import boolean_var
from utils import to_boolean
from torch.autograd import Function


#Need to convert weights to boolean
def binarize(inp):
    output = inp.new(inp.size())
    output[inp >= 0] = 1
    output[inp < 0] = -1
    return output

# It is a sign constraint which is sign of the sum of the partials
# of the matrix product
def add_sign_constraint(H, count, partial_poly, output_bool, lam, k_layer, j):
    # FIXME check if can be simplified since some vars are constant
    if count == 3:
        aux = 1
        aux_1 = boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux_2 = output_bool
        H.add_constraint_eq_zero(partial_poly - aux_1 - 2 * aux_2, lam=lam)
    elif count == 7:
        aux = 1
        aux_1 = boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 2
        aux_2 = boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux_4 = output_bool
        H.add_constraint_eq_zero(partial_poly - aux_1 - 2 * aux_2 - 4 * aux_4,
                                 lam=lam)
    elif count == 15:
        aux = 1
        aux_1 = boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 2
        aux_2 = boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 4
        aux_4 = boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux_8 = output_bool
        H.add_constraint_eq_zero(partial_poly - aux_1 - 2 * aux_2 - 4 * aux_4
                                 - 8 * aux_8, lam=lam)
    elif count == 31:
        aux = 1
        aux_1 = boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 2
        aux_2 = boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 4
        aux_4 = boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 8
        aux_8 = boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux_16 = output_bool
        H.add_constraint_eq_zero(partial_poly - aux_1 - 2 * aux_2 - 4 * aux_4
                                 - 8 * aux_8 - 16 * aux_16, lam=lam)
    elif count == 63:
        aux = 1
        aux_1 = boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 2
        aux_2 = boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 4
        aux_4 = boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 8
        aux_8 = boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 16
        aux_16 = boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux_32 = output_bool
        H.add_constraint_eq_zero(partial_poly - aux_1 - 2 * aux_2 - 4 * aux_4
                                 - 8 * aux_8 - 16 * aux_16 - 32 * aux_32,
                                 lam=lam)
    elif count == 127:
        aux = 1
        aux_1 = boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 2
        aux_2 = boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 4
        aux_4 = boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 8
        aux_8 = boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 16
        aux_16 = boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 32
        aux_32 = boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux_64 = output_bool
        H.add_constraint_eq_zero(partial_poly - aux_1 - 2 * aux_2 - 4 * aux_4
                                 - 8 * aux_8 - 16 * aux_16 - 32 * aux_32
                                 - 64 * aux_64, lam=lam)
    elif count == 255:
        aux = 1
        aux_1 = qv.boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 2
        aux_2 = qv.boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 4
        aux_4 = qv.boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 8
        aux_8 = qv.boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 16
        aux_16 = qv.boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 32
        aux_32 = qv.boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 64
        aux_64 = qv.boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux_128 = output_bool
        H.add_constraint_eq_zero(partial_poly - aux_1 - 2 * aux_2 - 4 * aux_4
                                 - 8 * aux_8 - 16 * aux_16 - 32 * aux_32
                                 - 64 * aux_64 - 128 * aux_128, lam=lam)
    elif count == 511:
        aux = 1
        aux_1 = qv.boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 2
        aux_2 = qv.boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 4
        aux_4 = qv.boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 8
        aux_8 = qv.boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 16
        aux_16 = qv.boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 32
        aux_32 = qv.boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 64
        aux_64 = qv.boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 128
        aux_128 = qv.boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux_256 = output_bool
        H.add_constraint_eq_zero(partial_poly - aux_1 - 2 * aux_2 - 4 * aux_4
                                 - 8 * aux_8 - 16 * aux_16 - 32 * aux_32
                                 - 64 * aux_64 - 128 * aux_128 - 256 * aux_256,
                                 lam=lam)
    elif count == 1023:
        aux = 1
        aux_1 = qv.boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 2
        aux_2 = qv.boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 4
        aux_4 = qv.boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 8
        aux_8 = qv.boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 16
        aux_16 = qv.boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 32
        aux_32 = qv.boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 64
        aux_64 = qv.boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 128
        aux_128 = qv.boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 256
        aux_256 = qv.boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux_512 = output_bool
        H.add_constraint_eq_zero(partial_poly - aux_1 - 2 * aux_2 - 4 * aux_4
                                 - 8 * aux_8 - 16 * aux_16 - 32 * aux_32
                                 - 64 * aux_64 - 128 * aux_128 - 256 * aux_256
                                 - 512 * aux_512,  lam=lam)
    else:
        raise NotImplementedError(f'count: {count}')


def setup_optim_model(sample_input_spin, sample_input_target, model, args):
    # Rahul: ONLY ZERO objective is tested
    LAMBDA = args.LAMBDA
    epsilon = args.epsilon
    sample_input_boolean = to_boolean(sample_input_spin)
    qubo_vars = {}  # dict of variables
    H = qv.PCBO()


    # Minimize Perturbation
    sum_taus = 0
    for i in args.pixels_to_perturb:
        qubo_vars[f'tau_{i}'] = qv.boolean_var(f'tau_{i}')
        sum_taus += qubo_vars[f'tau_{i}']

    # TODO: Implement other objectives
    if args.objective == 'output':
        raise NotImplementedError('output')

    # Bound the number of perturbations less than epsilon
    if args.include_perturbation_bound_constraint: 
        H.add_constraint_lt_zero(sum_taus - epsilon,
                                 LAMBDA['perturbation_bound_constraint'])

    fc_in = sample_input_boolean

    for idx, layer in enumerate(model.named_parameters()):
        param = layer[1]
        if idx == len(list(model.modules()))-1:
            break
        # loop over output dimension of layer
        for j in range(param.size()[0]):
            sum_partials = 0
            count_partials = 0
            # loop over input dimension of layer
            for i in range(param.size()[1]):
                weight = to_boolean(binarize(param[j][i]))
                qubo_vars[
                    f'partial_matrix_product_{idx}_{i}_{j}'
                    ] = qv.boolean_var(f'partial_matrix_product_{idx}_{i}_{j}')
                if idx == 0: # First Layer Simplification
                    if i not in args.pixels_to_perturb:  # tau_i is 0 (Seyran) # Leaving unperturbed pixels
                        if fc_in[i] == 1:
                            if weight == 0:
                                qubo_vars[
                                    f'partial_matrix_product_{idx}_{i}_{j}'
                                    ] = 0
                            elif weight == 1:
                                qubo_vars[
                                    f'partial_matrix_product_{idx}_{i}_{j}'
                                    ] = 1
                        elif fc_in[i] == 0:
                            if weight == 0:
                                qubo_vars[
                                    f'partial_matrix_product_{idx}_{i}_{j}'
                                    ] = 1
                            elif weight == 1:
                                qubo_vars[
                                    f'partial_matrix_product_{idx}_{i}_{j}'
                                    ] = 0
                    else:
                        # TODO: if i not in range(3, 6) then assume tau_i is
                        # zero and subsequent vars (Seyran) 
                        # Adding Perturbation with First layer simplification
                        if fc_in[i] == 1:
                            if weight == 0:
                                H.add_constraint_eq_BUFFER(
                                    qubo_vars[
                                        f'partial_matrix_product_{idx}_{i}_{j}'
                                        ],
                                    qubo_vars[f'tau_{i}'],
                                    lam=LAMBDA['hard_constraints']
                                )
                            elif weight == 1:
                                H.add_constraint_eq_NOT(
                                    qubo_vars[
                                        f'partial_matrix_product_{idx}_{i}_{j}'
                                        ],
                                    qubo_vars[f'tau_{i}'],
                                    lam=LAMBDA['hard_constraints']
                                )
                        elif fc_in[i] == 0:
                            if weight == 0:
                                H.add_constraint_eq_NOT(
                                    qubo_vars[
                                        f'partial_matrix_product_{idx}_{i}_{j}'
                                        ],
                                    qubo_vars[f'tau_{i}'],
                                    lam=LAMBDA['hard_constraints']
                                )
                            elif weight == 1:
                                H.add_constraint_eq_BUFFER(
                                    qubo_vars[
                                        f'partial_matrix_product_{idx}_{i}_{j}'
                                        ],
                                    qubo_vars[f'tau_{i}'],
                                    lam=LAMBDA['hard_constraints']
                                )
                else: # General Layer ==> Reasoning : Since weights are fixed XNOR is not required
                    if weight == 0:
                        H.add_constraint_eq_NOT(
                            qubo_vars[f'partial_matrix_product_{idx}_{i}_{j}'],
                            fc_in[i],
                            lam=LAMBDA['hard_constraints']
                        )
                    elif weight == 1:
                        H.add_constraint_eq_BUFFER(
                            qubo_vars[f'partial_matrix_product_{idx}_{i}_{j}'],
                            fc_in[i],
                            lam=LAMBDA['hard_constraints']
                        )
            
                sum_partials += qubo_vars[
                    f'partial_matrix_product_{idx}_{i}_{j}']
                count_partials += 1
            qubo_vars[
            f'matrix_product_{idx}_{j}'
            ] = qv.boolean_var(f'matrix_product_{idx}_{j}')

            # Output is decided layer wise and end is just 
            if idx == len(list(model.modules()))-2:
                output_bool = int(not sample_input_target)
            else:
                output_bool = qubo_vars[f'matrix_product_{idx}_{j}']

            add_sign_constraint(H, count_partials,
                                sum_partials,
                                output_bool=output_bool,
                                lam=LAMBDA['hard_constraints'],
                                k_layer=idx,
                                j=j)

        fc_in = [qubo_vars[f'matrix_product_{idx}_{j}'] for j in range(param.size()[0])]

    ordered_variables = list(H.convert_solution([0]*len(H.to_qubo().variables)).keys())

    return H, ordered_variables
