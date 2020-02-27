from __main__ import *

def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm


def gradient_single_example(X, y, w):
    l1 = np.matmul(X, w) * y
    yx = y * X
    return - sigmoid(-l1) * yx

def gradient_single_node(X, y, w):
    return np.mean(f_gradient(X, y, w), axis=0)

def f_gradient(X, y, w):
    yX = y[:, np.newaxis] * X
    l1 = np.matmul(yX, w)
    return -sigmoid(-l1)[:, np.newaxis] * yX


def reg_gradient(w, lmbda):
    return 2 * lmbda * w


def gradient(X, y, w):
    # return reg_gradient(w, lmbda) + np.mean(f_gradient(X, y, w), axis=0)
    return np.mean(f_gradient(X, y, w), axis=0)


def gradient_full(X, y, w):
    # return reg_gradient(w, lmbda) + f_gradient(X, y, w)
    return f_gradient(X, y, w)


def cost(X, y, w):

    l1 = np.matmul(X, w) * y
    return np.mean(np.log(1 + np.exp(-l1)), axis=0)

def binary_classification_cost(X, y, w):
    return f1_score(y_true = y, y_pred = np.sign(np.matmul(X, w)), average='micro')

def ax_modifier(ax, legend_loc, ncol, xlabel, ylabel, title=None):
    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    no_curves = len(ax.lines)
    ymin = min(ymin, min([min(ax.lines[i].get_ydata()) for i in range(no_curves)]))
    ymax = max(ymax, max([max(ax.lines[i].get_ydata()) for i in range(no_curves)]))
    xmax = max(xmax, max([max(ax.lines[i].get_xdata()) for i in range(no_curves)]))
    ax.legend(loc=legend_loc)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_xticks(np.arange(0, xmax+1, step=round(xmax/15)))
    ax.legend(ncol=ncol)
    if title is not None:
        ax.set_title(title)
    #if not(np.isinf([ymin, ymax]).all()) and (not(np.isnan([ymin, ymax]).all())):  # if boundaries are defined
    #    ax.set_ylim((0.98*ymin, 1.02*ymax))


def random_sampler(N, batch=1, buffersize=1000):
    """
    A generator of random indices from 0 to N.
    params:
    N: upper bound of the indices
    batch: Number of indices to return per iteration
    buffersize: Number of numbers to generate per batch
                (this is only a computational nicety)
    """

    S = int(np.ceil(buffersize / batch))

    while True:
        buffer = np.random.randint(N, size=(S, batch))
        for i in range(S):
            yield buffer[i]

def arbitrary_sampler(N, prob_vector, batch=1, buffersize=1000):
    """
    A generator of random indices from 0 to N.
    params:
    N: upper bound of the indices
    batch: Number of indices to return per iteration
    prob_vector: probability of taking index [i] is prob_vector[i]
    buffersize: Number of numbers to generate per batch
                (this is only a computational nicety)
    """

    S = int(np.ceil(buffersize / batch))

    while True:
        buffer = np.zeros((S, batch), dtype=np.int64)
        for s in range(S):
            for b in range(batch):
                buffer[s][b] = np.where(np.random.multinomial(1, prob_vector) == 1)[0]

        for s in range(S):
            yield buffer[s]


def minibatch_sampler(N, minibatch_size, prob_vector, epoch_size, buffersize=1000):
    """
    A generator of random indices from 0 to N.
    params:
    N: upper bound of the indices
    minibatch_size: Number of indices to return per iteration
    epoch_size: number of vectors (set of indices) to return per epoch
    prob_vector: sampling probability to return numbers
    buffersize: Number of numbers to generate per batch
                (this is only a computational nicety)
    """

    S = int(np.ceil(buffersize / minibatch_size*epoch_size))
    
    buffer = [[0 for i in range(epoch_size)] for j in range(S)] 

    while True:
        for j in range(epoch_size):
            for i in range(S):
                idx = np.where(np.random.random_sample(N) < prob_vector)[0]
                if np.isnan(idx).all():
                    idx = np.array([0])
                
                buffer[i][j] = idx
        for i in range(S):
            yield buffer[i]
            

def GD(X, y, w, learning_rate=0.1, iterations=1000):
    for iteration in range(iterations):
        grad = gradient(X, y, w)
        w = w - learning_rate * grad

        yield w


def SGD(X, y, w, learning_rate, batch_size=1):
    N, D = X.shape
    sampler = random_sampler(N, batch_size)

    for ix in sampler:
        grad = gradient(X[ix], y[ix], w)
        w = w - learning_rate * grad
        yield w


def SVRG(X, y, w, learning_rate, epoch_size, N):
    """
    Stochastic variance reduced gradient
    """

    sampler = random_sampler(N, epoch_size)

    for epoch in sampler:
        full_grad = np.array([gradient(X[i], y[i], w) for i in range(N)])
        mean_grad = np.mean(full_grad, axis=0)

        for ix in epoch:
            grad = gradient_single_node(X[ix], y[ix], w)
            w = w - learning_rate * (grad - full_grad[ix] + mean_grad)

        yield w

def SVRG_AS(X, y, w, learning_rate, epoch_size, N, prob_vector):
    """
    Stochastic variance reduced gradient with arbitrary sampling
    """

    sampler = arbitrary_sampler(N, prob_vector, epoch_size)

    for epoch in sampler:
        full_grad = np.array([gradient(X[i], y[i], w) for i in range(N)])
        mean_grad = np.mean(full_grad, axis=0)

        for ix in epoch:
            grad = gradient_single_node(X[ix], y[ix], w)
            w = w - learning_rate * (grad/(N*prob_vector[ix]) - full_grad[ix]/(N*prob_vector[ix]) + mean_grad)

        yield w


def minibatch_SVRG(X, y, w, learning_rate, epoch_size, N, minibatch_size):
    """
    Stochastic variance reduced gradient with minibatch updates
    """

    sampler = minibatch_sampler(N, minibatch_size, prob_vector=np.ones(N)/N*minibatch_size, epoch_size=epoch_size)
    for epoch in sampler:
        full_grad = np.array([gradient(X[i], y[i], w) for i in range(N)])
        mean_grad = np.mean(full_grad, axis=0)
        for minibatch in epoch:
            grad = np.mean(np.array([gradient_single_node(X[i], y[i], w) for i in minibatch]), axis=0)
            grad_hat = np.mean(np.array([full_grad[i] for i in minibatch]), axis=0)
            w = w - learning_rate * (grad - grad_hat + mean_grad)

        yield w

def minibatch_SVRG_AS(X, y, w, learning_rate, epoch_size, N, prob_vector, minibatch_size):
    """
    Stochastic variance reduced gradient with arbitrary sampling and minibatch updates
    """

    sampler = minibatch_sampler(N, minibatch_size, prob_vector, epoch_size)

    for epoch in sampler:
        full_grad = np.array([gradient(X[i], y[i], w) for i in range(N)])
        mean_grad = np.mean(full_grad, axis=0)
        for minibatch in epoch:
            grad = np.sum(np.array([gradient_single_node(X[i], y[i], w)/(N*prob_vector[i]) for i in minibatch]), axis=0)
            grad_hat = np.sum(np.array([full_grad[i]/(N*prob_vector[i]) for i in minibatch]), axis=0)
            w = w - learning_rate * (grad - grad_hat + mean_grad)

        yield w


def initialize_w(N):
    return np.random.randn(N)


def loss(X, y, w):
    objective_loss = cost(X, y, w)
    f1_score = binary_classification_cost(X, np.sign(y), w)
    return objective_loss, f1_score


def iterate(opt, X_train_partitioned, y_train_partitioned, X_train, y_train, X_test, y_test, w_0,
            iterations=100, inner=1, name="NoName", printout=True):
    """
    This function takes an optimizer and returns a loss history for the
    training and test sets.
    """

    loss_hist_train, train_f1_score = loss(X_train, y_train, w_0)
    loss_hist_test, test_f1_score = loss(X_test, y_test, w_0)
    mean_grad_norm = np.linalg.norm(np.mean(gradient_full(X_train, y_train, w_0), axis=0), 2)

    ws = [w_0]
    clock = [0]

    start = time.time()
    for iteration in range(iterations):
        # print("\n ------Iteration {} is started-----".format(iteration))
        for _ in range(inner):
            w = next(opt)
        clock.append(time.time() - start)
        ws.append(w)

    # for iteration, w in enumerate(ws):
        train_loss, train_f1_score_new = loss(X_train, y_train, w)
        loss_hist_train = np.append(loss_hist_train, train_loss)
        train_f1_score = np.append(train_f1_score, train_f1_score_new)

        test_loss, test_f1_score_new = loss(X_test, y_test, w)
        loss_hist_test = np.append(loss_hist_test, test_loss)
        test_f1_score = np.append(test_f1_score, test_f1_score_new)
        grad_norm_new = np.linalg.norm(np.mean(gradient_full(X_train, y_train, w), axis=0), 2)
        mean_grad_norm = np.append(mean_grad_norm, grad_norm_new)

        if printout:
            print('{}; Iter = {:02}; Objective(train) = {:05.3f}; Objective(test) = {:05.3f}; F1score(train) = {:05.3f}; F1score(test) = {:05.3f}'.format(name, iteration, train_loss, test_loss, train_f1_score_new, test_f1_score_new))
        # print('Current solution = {}'.format(w.round(decimals=3)))
        sys.stdout.flush()

    return ws[-1], loss_hist_train, loss_hist_test, train_f1_score, test_f1_score, clock, mean_grad_norm

def QM_SVRG_A(X, y, w, learning_rate, epoch_size,
              parameter_quantization_opt, grad_quantization_opt, strong_convexity_param):

    """
    Stochastic variance reduced gradient with fixed quantization grids
    """

    no_samples, dimension = X_train.shape
    sampler = random_sampler(no_samples, epoch_size)

    for batch in sampler:
        w_old = np.copy(w)
        full_grad = gradient_full(X, y, w)
        mean_grad = np.mean(full_grad, axis=0)

        quantization = {'parameter':{'no_bits':{}, 'center':{}, 'radius':{}},
                        'gradient':{'no_bits':{}, 'center':{}, 'radius':{}}}

        # The following code de-activates quantization of the initial gradient reporting at the outer-loop
        if isinstance(parameter_quantization_opt['no_bits'], int):
            quantization['parameter']['no_bits'] = np.tile([parameter_quantization_opt['no_bits']], dimension)
        if isinstance(grad_quantization_opt['no_bits'], int):
            quantization['gradient']['no_bits'] = np.tile([grad_quantization_opt['no_bits']], dimension)


        quantization['gradient']['center'] = mean_grad
        quantization['gradient']['radius'] = np.tile(2 * np.linalg.norm(mean_grad, 2), dimension)
        quantization['parameter']['center'] = w_old
        quantization['parameter']['radius'] = np.tile(2 * np.linalg.norm(mean_grad, 2) / strong_convexity_param, dimension)

        # print('-- Grid_center is {} \n -- Quantization radius is {}'.format(grid_center.round(decimals=3), quantization_radius[0].round(decimals=3)))
        #        print('grads are ', np.mean(full_grad, axis=0), '--norm is ', np.linalg.norm(np.mean(full_grad, axis=0)), '--strong_convexity_param is ', strong_convexity_param)

        for ix in batch:
            grad = gradient_single(X[ix], y[ix], w)
            quantized_grad = vector_quantization(original_vector=grad,
                                                 quantization_radius=quantization['gradient']['radius'],
                                                 grid_center=quantization['gradient']['center'],
                                                 no_bits=quantization['gradient']['no_bits'])

            w = w - learning_rate * (quantized_grad - full_grad[ix] + mean_grad)
            w = vector_quantization(original_vector=w,
                                    quantization_radius=quantization['parameter']['radius'],
                                    grid_center=quantization['parameter']['center'],
                                    no_bits=quantization['parameter']['no_bits'])

        mean_grad_new = np.mean(gradient_full(X, y, w), axis=0)
        if np.linalg.norm(mean_grad, 2) <= np.linalg.norm(mean_grad_new, 2):
            # print('\n-NoU-', np.linalg.norm(mean_grad, 2), np.linalg.norm(mean_grad_new, 2), w_old, w, '\n')
            w = np.copy(w_old)

        # else:
            # print('-U-', np.linalg.norm(mean_grad, 2), np.linalg.norm(mean_grad_new, 2), w_old, w)

        yield w


def cost_SVRG_AS(cost_vector, prob_vector):
    return np.sum(cost_vector * prob_vector)


def smoothness_UB(X_train_partitioned, N):
    return np.array([np.sum(X_train_partitioned[i]**2)/(4*X_train_partitioned[i].shape[0]) for i in range(N)])


def T_LB(mu, alpha, L, sigma_max):
    return 1/(mu*alpha*(sigma_max - 2*L*alpha*sigma_max-2*L*alpha))


def sigma_bound(mu, T, alpha, L):
    return (1/(mu*T*alpha)+ 2*L*alpha)/(1- 2*L*alpha)


def p_LB_stragglers(smoothness, alpha, N):
    return (4*smoothness/N) * max(alpha, 1/(4*max(smoothness)))


def p_LB_congestion(smoothness, alpha, N, K, T, mu, eps1, Delta0):
    eps2 = ((eps1/Delta0)**(1/K))/(1+(eps1/Delta0)**(1/K))*(1+1/(mu*T*alpha)) - 1/(mu*T*alpha)
    p = (2*smoothness/N) * max(alpha/eps2, 1/(2*max(smoothness)))
    p[p > 1] = 1
    return p


def alpha_UB(L, sigma_max):
    return sigma_max/(2*L*(1+sigma_max))



def optimizer_binary_classification(target_dataset, X_train_partitioned, y_train_partitioned, X_train, y_train,
                                                    X_test, y_test, w_0,
                                                    hyper_parameters, multi_class=False, printout=True):
    optimizers = [
            {
                    "opt": SVRG(X=X_train_partitioned, y=y_train_partitioned, w=w_0,
                                learning_rate=hyper_parameters['learning_rate'],
                                epoch_size=hyper_parameters['SVRG_epoch_size'],
                                N=hyper_parameters['no_nodes']),
                    "name": "SVRG",
                    "inner": 1
            },
            {
                    "opt": SVRG_AS(X=X_train_partitioned, y=y_train_partitioned, w=w_0,
                                  learning_rate=hyper_parameters['learning_rate'],
                                  epoch_size=hyper_parameters['SVRG_epoch_size'],
                                  N=hyper_parameters['no_nodes'],
                                  prob_vector=hyper_parameters['sampling_probability']),
                    "name": "SVRG-AS",
                    "inner": 1
            },
    ]

    outputs = {optimizers[i]['name']: {'optimal_parameter': {}, 'training_loss': {}, 'training_f1_score': {}, 'training_mean_grad_norm': {}}
               for i in np.arange(len(optimizers))}

    fig, ax = plt.subplots(3, 1, figsize=(13, 12))

    for opt in optimizers:

        w, loss_hist_train, loss_hist_test, train_f1_score, test_f1_score, clock, mean_grad_norm = iterate(
                opt['opt'],
                X_train_partitioned, y_train_partitioned, X_train, y_train, X_test, y_test, w_0,
                iterations=hyper_parameters['iterations'], inner=opt['inner'],
                name=opt['name'], printout=printout)
        outputs[opt['name']]['optimal_parameter'] = w
        outputs[opt['name']]['training_loss'] = loss_hist_train
        outputs[opt['name']]['training_f1_score'] = train_f1_score
        outputs[opt['name']]['training_mean_grad_norm'] = mean_grad_norm

        color = next(ax[0]._get_lines.prop_cycler)['color']
        iterations_axis = range(0, hyper_parameters['iterations'] + 1)
        ax[0].plot(iterations_axis, loss_hist_train,
               label="Train loss ({})".format(opt['name']), linestyle="-", color=color)

        ax[0].plot(iterations_axis, loss_hist_test,
               label="Test loss ({})".format(opt['name']), linestyle="--", color=color)

        ax[1].plot(iterations_axis, mean_grad_norm,
               label="Training ({})".format(opt['name']), linestyle="-", color=color)

        # ax[2].plot(iterations_axis, train_f1_score,
        #       label="Train F1 score ({})".format(opt['name']), linestyle="-", color=color)

        ax[2].plot(iterations_axis, test_f1_score,
               label="{}".format(opt['name']), linestyle="-", color=color)

        # ax[3].plot(clock, loss_hist_train,
        #           label="Train loss ({})".format(opt['name']), linestyle="-", color=color)
        # ax[3].plot(clock, loss_hist_test,
        #           label="Test loss ({})".format(opt['name']), linestyle="--", color=color)

    ax_modifier(ax=ax[0], legend_loc="upper right", ncol=2, xlabel="Iteration", ylabel="Loss",
                title="Performance Comparison of various algorithms")
    ax_modifier(ax=ax[1], legend_loc="upper right", ncol=2, xlabel="Iteration", ylabel="Gradient norm (training)")
    ax_modifier(ax=ax[2], legend_loc="lower right", ncol=2, xlabel="Iteration", ylabel="F1 score (test)")
    # ax_modifier(ax=ax[3], legend_loc="upper right", xlabel="Time in seconds", ylabel="Loss")

    if multi_class:
        fig_name = './TestResults/noQuant_allAlg_'+target_dataset+'_Class'+str(y_train[0])
    else:
        fig_name = './TestResults/'+target_dataset+'_BinaryClassification'

    plt.savefig(fig_name+'.png')
    plt.savefig(fig_name+'.pdf')
    matplotlib2tikz.save(fig_name+'.tex')

    return outputs


def optimizer_multiclass_classification(target_dataset, X_train_partitioned, y_train_partitioned, X_train, y_train,
                                        X_test, y_test, w_0,
                                        hyper_parameters, class_idx, printout=True):

    optimizers = [
            {
                "opt": SVRG(X=X_train_partitioned, y=y_train_partitioned, w=w_0,
                            learning_rate=hyper_parameters['learning_rate'],
                            epoch_size=hyper_parameters['SVRG_epoch_size'],
                            N=hyper_parameters['no_nodes']),
                "name": "SVRG",
                "inner": 1
            },
            {
                "opt": SVRG_AS(X=X_train_partitioned, y=y_train_partitioned, w=w_0,
                               learning_rate=hyper_parameters['learning_rate'],
                               epoch_size=hyper_parameters['SVRG_epoch_size'],
                               N=hyper_parameters['no_nodes'],
                               prob_vector=hyper_parameters['sampling_probability']),
                "name": "SVRG-AS",
                "inner": 1
            },
    ]
    outputs = {optimizers[i]['name']: {'optimal_parameter': {}, 'training_loss': {}, 'training_f1_score': {},
                                       'training_mean_grad_norm': {}}
               for i in np.arange(len(optimizers))}


    fig, ax = plt.subplots(3, 1, figsize=(13, 12))

    for opt in optimizers:

        w, loss_hist_train, loss_hist_test, train_f1_score, test_f1_score, clock, mean_grad_norm = iterate(
                opt['opt'],
                X_train_partitioned, y_train_partitioned, X_train, y_train, X_test, y_test, w_0,
                iterations=hyper_parameters['iterations'], inner=opt['inner'],
                name=opt['name'], printout=printout)

        outputs[opt['name']]['optimal_parameter'] = w
        outputs[opt['name']]['training_loss'] = loss_hist_train
        outputs[opt['name']]['training_f1_score'] = train_f1_score
        outputs[opt['name']]['training_mean_grad_norm'] = mean_grad_norm

        color = next(ax[0]._get_lines.prop_cycler)['color']

        iterations_axis = range(0, hyper_parameters['iterations'] + 1)
        ax[0].plot(iterations_axis, loss_hist_train,
               label="{}".format(opt['name']), linestyle="-", color=color)

        ax[1].plot(iterations_axis, mean_grad_norm,
               label="{}".format(opt['name']), linestyle="-", color=color)

        ax[2].plot(iterations_axis, test_f1_score,
               label="{}".format(opt['name']), linestyle="-", color=color)

        # ax[3].plot(clock, loss_hist_train,
        #           label="Train loss ({})".format(opt['name']), linestyle="-", color=color)
        # ax[3].plot(clock, loss_hist_test,
        #           label="Test loss ({})".format(opt['name']), linestyle="--", color=color)

    ax_modifier(ax=ax[0], legend_loc="upper right", ncol=2, xlabel="Iteration", ylabel="Loss (training)",
                title="Performance Comparison of various algorithms")
    ax_modifier(ax=ax[1], legend_loc="upper right", ncol=2, xlabel="Iteration", ylabel="Gradient norm (training)")
    ax_modifier(ax=ax[2], legend_loc="lower right", ncol=2, xlabel="Iteration", ylabel="F1 score (test)")
    # ax_modifier(ax=ax[3], legend_loc="upper right", xlabel="Time in seconds", ylabel="Loss")

    fig_name = './TestResults/_'+target_dataset+'_Class'+str(class_idx)

    plt.savefig(fig_name+'.png')
    plt.savefig(fig_name+'.pdf')
    #matplotlib2tikz.save(fig_name+'.tex')

    return outputs


def cost_evaluation_stragglers(target_dataset, optimizers, hyper_parameters, no_class, plot=True):

    no_scenarios = len(hyper_parameters['cost_model'])
    if plot:
        fig, ax = plt.subplots(no_scenarios, 1, figsize=(13, 16))
    i = 0
    cost = {'scenario_'+str(i+1): {j: {} for j in optimizers} for i in range(len(hyper_parameters['cost_model']))}
    iterations_axis = range(0, hyper_parameters['iterations'] + 1)
    for cost_vector in hyper_parameters['cost_model']:
        for algorithm_name in optimizers:
            
            outer_loop_cost = np.sum(cost_vector['values'])

            if (no_class == 2) and (algorithm_name == 'SVRG'):
                cost_per_iter = outer_loop_cost + hyper_parameters['SVRG_epoch_size']*np.mean(cost_vector['values'])
            elif (no_class > 2) and (algorithm_name == 'SVRG'):
                cost_per_iter = no_class *(outer_loop_cost + hyper_parameters['SVRG_epoch_size'] * np.mean(cost_vector['values']))
            elif (no_class == 2) and (algorithm_name == 'SVRG-AS'):
                cost_per_iter = outer_loop_cost + np.mean(hyper_parameters['sampling_probability']*cost_vector['values'])* hyper_parameters['SVRG_epoch_size']/2
            elif (no_class > 2) and (algorithm_name == 'SVRG-AS'):
                cost_per_iter = no_class*(outer_loop_cost + np.mean(hyper_parameters['sampling_probability']*cost_vector['values']) * hyper_parameters['SVRG_epoch_size']/2)
            elif (no_class == 2) and (algorithm_name == 'GD'):
                cost_per_iter = outer_loop_cost 
            elif (no_class > 2) and (algorithm_name == 'GD'):
                cost_per_iter = no_class * outer_loop_cost 
            else:
                cost_per_iter = 0

            cost_array = np.array([0] + [cost_per_iter for i in range(hyper_parameters['iterations'])])
            cost['scenario_' + str(i + 1)][algorithm_name] = np.cumsum(cost_array)

            if plot:
                color = next(ax[0]._get_lines.prop_cycler)['color']
                ax[i].plot(iterations_axis, cost['scenario_'+str(i+1)][algorithm_name],
                       label="{} (scenario {})".format(algorithm_name, i+1), linestyle="-", color=color)

        if plot:
            ax_modifier(ax=ax[i], legend_loc="upper right", ncol=2, xlabel="Iteration", ylabel="Cost")
        i += 1

    if plot:
        fig_name = './TestResults/Cost_stragglers_'+target_dataset
        plt.savefig(fig_name+'.png')
        plt.savefig(fig_name+'.pdf')
        matplotlib2tikz.save(fig_name+'.tex')

    return cost


def cost_function_congestion(hyper_parameters, plot=True):
    no_scenarios = len(hyper_parameters['cost_model'])
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(13, 5))

    x_axis = np.arange(0.000001, 101, 0.2)

    i = 0
    y_axis = [[0], [0], [0]]
    for rate_model in hyper_parameters['cost_model'][0]['rate_models']:
        r0 = rate_model[0]
        r1 = rate_model[1]
        y_axis[i] = 1 / (r0 * x_axis * np.exp(-x_axis / r1))
        i += 1
    if plot:
        labels = ["(r_0, r_1)={}".format(hyper_parameters['cost_model'][0]['rate_models'][0]),
                  "(r_0, r_1)={}".format(hyper_parameters['cost_model'][0]['rate_models'][1]),
                  "(r_0, r_1)={}".format(hyper_parameters['cost_model'][0]['rate_models'][2])]

        for y_arr, label in zip(y_axis, labels):
            plt.plot(x_axis, y_arr, label=label)
            # plt.yscale('symlog')

        ax_modifier(ax=ax, legend_loc="upper left", ncol=1, xlabel="b", ylabel="Cost")
        ax.set_ylim((0, 10))
        ax.set_xlim((0, 50))

    if plot:
        fig_name = './TestResults/minibatch_cost_models'
        plt.savefig(fig_name + '.png')
        plt.savefig(fig_name + '.pdf')
        matplotlib2tikz.save(fig_name + '.tex')

    results = {'minibatch_size': x_axis, 'cost': y_axis}
    return results


def optimizer_multiclass_classification_minibatch(target_dataset, X_train_partitioned, y_train_partitioned, X_train, y_train,
                                                  X_test, y_test, w_0,
                                                  hyper_parameters, class_idx, printout=True):

    optimizers = [
            {
                "opt": minibatch_SVRG(X=X_train_partitioned, y=y_train_partitioned, w=w_0,
                                      learning_rate=hyper_parameters['learning_rate'],
                                      epoch_size=hyper_parameters['SVRG_epoch_size'],
                                      N=hyper_parameters['no_nodes'],
                                      minibatch_size=hyper_parameters['minibatch_size']),
                "name": "m-SVRG",
                "inner": 1
            },
            {
                "opt": minibatch_SVRG_AS(X=X_train_partitioned, y=y_train_partitioned, w=w_0,
                                         learning_rate=hyper_parameters['learning_rate'],
                                         epoch_size=hyper_parameters['SVRG_epoch_size'],
                                         N=hyper_parameters['no_nodes'],
                                         prob_vector=hyper_parameters['sampling_probability'],
                                         minibatch_size=hyper_parameters['minibatch_size']),
                "name": "m-SVRG-AS",
                "inner": 1
            },
    ]
    outputs = {optimizers[i]['name']: {'optimal_parameter': {}, 'training_loss': {}, 'training_f1_score': {},
                                       'training_mean_grad_norm': {}}
               for i in np.arange(len(optimizers))}


    fig, ax = plt.subplots(3, 1, figsize=(13, 12))

    for opt in optimizers:

        w, loss_hist_train, loss_hist_test, train_f1_score, test_f1_score, clock, mean_grad_norm = iterate(
                opt['opt'],
                X_train_partitioned, y_train_partitioned, X_train, y_train, X_test, y_test, w_0,
                iterations=hyper_parameters['iterations'], inner=opt['inner'],
                name=opt['name'], printout=printout)

        outputs[opt['name']]['optimal_parameter'] = w
        outputs[opt['name']]['training_loss'] = loss_hist_train
        outputs[opt['name']]['training_f1_score'] = train_f1_score
        outputs[opt['name']]['training_mean_grad_norm'] = mean_grad_norm

        color = next(ax[0]._get_lines.prop_cycler)['color']

        iterations_axis = range(0, hyper_parameters['iterations'] + 1)
        ax[0].plot(iterations_axis, loss_hist_train,
               label="{}".format(opt['name']), linestyle="-", color=color)

        ax[1].plot(iterations_axis, mean_grad_norm,
               label="{}".format(opt['name']), linestyle="-", color=color)

        ax[2].plot(iterations_axis, test_f1_score,
               label="{}".format(opt['name']), linestyle="-", color=color)

        # ax[3].plot(clock, loss_hist_train,
        #           label="Train loss ({})".format(opt['name']), linestyle="-", color=color)
        # ax[3].plot(clock, loss_hist_test,
        #           label="Test loss ({})".format(opt['name']), linestyle="--", color=color)

    ax_modifier(ax=ax[0], legend_loc="upper right", ncol=2, xlabel="Iteration", ylabel="Loss (training)",
                title="Performance Comparison of various algorithms")
    ax_modifier(ax=ax[1], legend_loc="upper right", ncol=2, xlabel="Iteration", ylabel="Gradient norm (training)")
    ax_modifier(ax=ax[2], legend_loc="lower right", ncol=2, xlabel="Iteration", ylabel="F1 score (test)")
    # ax_modifier(ax=ax[3], legend_loc="upper right", xlabel="Time in seconds", ylabel="Loss")

    fig_name = './TestResults/_'+target_dataset+'_Class'+str(class_idx)

    plt.savefig(fig_name+'.png')
    plt.savefig(fig_name+'.pdf')
    #matplotlib2tikz.save(fig_name+'.tex')

    return outputs
