import numpy as np
from scipy import sparse 
from random import choices
from scipy import sparse 
from scipy import integrate
from scipy.optimize import brentq
from numpy.linalg import svd 
#from two_area_network import *  



class line_attractor:
    def __init__(self, modelparams, task):
        np.random.seed(modelparams['seed'])
        self.N = modelparams['N_pfc']
        self.dim = modelparams['dim']
        self.g = modelparams['g']
        self.std_priv = modelparams['sigma_noise_pfc']
        self.tau_noise = modelparams['tau_noise_pfc']
        self.dt = modelparams['dt']
        self.make_matrix()
        self.task = task

    def make_matrix(self):
        K = self.N
        rand = np.random.normal(0, 1, (self.N, self.N))
        #dil = np.random.binomial(1, self.c, (self.N, self.N))
        mat = rand * (self.g/np.sqrt(K))-np.eye(self.N)  
        u, s, vh = svd(mat)
        null_mat = np.zeros((self.N, self.N))
        for l in range(self.dim):
            null_mat += s[self.N - 1 - l] * np.outer(u[:, self.N - 1 - l], vh[self.N - 1 - l, :])
        self.matrix = mat -  null_mat
        self.null = u[:,-self.dim:].transpose()
        self.rank = u[:,0:self.N - self.dim].transpose()
        #self.null = vh[-self.dim:, :]
        #self.rank = vh[0:self.N - self.dim, :]
    
    def tf_sig(self, h):
        '''tanh transfer function'''
        tf =  np.tanh(h)
        #tf = h
        return tf

    def _overlaps(self, r):
        '''compute overlaps'''
        r_norm = r/np.linalg.norm(r)
        ovs = np.einsum('ij,j->i', self.null, r)
        return ovs
    
    def noise(self, x):
        gauss = np.random.normal(0, 1, self.N)
        const = np.sqrt((2 * self.std_priv**2 * self.tau_noise)/self.dt)
        noise = const * gauss
        return -x/self.tau_noise + noise

    def field(self, u, x, t):
        #stim = self.stim.stimulus(t, null = self.stim_type)
        stimulus = self.task.input_current(t)
        stim = stimulus['pfc']
        #self.tf_sig(u)
        return self.matrix.dot(u) + stim + x



class WongWang(object):
    ''' Wong-Wang decision area'''
    def __init__(self, params):
        self.dt = params['dt']
        #----------------------------------
        # Wong&Wang model
        self.A_t = params['A_t'] #Hz/nA
        self.B_t = params['B_t'] #Hz
        self.D_t = params['D_t'] #s
        self.tau_nmda = params['tau_nmda'] #s
        self.gamma = params['gamma']
        self.i_0 = params['i_0'] #nA
        self.tau_ampa = params['tau_ampa'] #ms
        self.sig_ampa = params['sig_ampa']#8#nA amplitude noise
        self.w_a_a = params['w_a_a']
        self.w_b_b = params['w_b_b']
        self.j_a_b = params['j_a_b']
        self.j_b_a = params['j_b_a']
        #-------------------------


    def t_fun_ww(self, current):
        '''transfer function, Wong&Wang, 2006'''
        num = self.A_t * current - self.B_t
        den = 1- np.exp(-self.D_t * num)
        return num/den


    def field_colored_noise(self, curr_a, curr_b):
        '''field of the noise to each population'''
        noise_a = np.random.normal(0, 1)
        noise_b = np.random.normal(0, 1)
        cons = np.sqrt(self.dt * (self.sig_ampa**2) * self.tau_ampa)/self.tau_ampa
        field_a = -curr_a * (self.dt/self.tau_ampa) + cons *  noise_a
        field_b = -curr_b * (self.dt/self.tau_ampa) + cons *  noise_b
        return field_a, field_b


    def field_one_area(self, s_a, s_b,  i_a, i_b):
        '''field large scale
        r_e_a: pop excitatory neurons a
        r_e_b: pop excitateory neurons b
        '''
        # local and long-range inputs
        curr_a =  self.w_a_a * s_a - self.j_a_b * s_b + i_a + self.i_0
        curr_b =  self.w_b_b * s_b - self.j_b_a * s_a + i_b + self.i_0
        #rates
        r_a = self.t_fun_ww(curr_a)
        r_b = self.t_fun_ww(curr_b)
        #fields
        f_a = -s_a/self.tau_nmda + (1 - s_a) * self.gamma * r_a
        f_b = -s_b/self.tau_nmda + (1 - s_b) * self.gamma * r_b
        return f_a, f_b, r_a, r_b

class NetworkDynamics:
    '''This class creates the connectivity matrix'''
    def __init__(self, modelparams, params_ww, task):
        #tranfer function and learning rule
        self.task = task
        np.random.seed()#randomizing the seed 
        self.dt = modelparams['dt'] # dt integration
        self.period = modelparams['period']
        self.amp_rpe = modelparams['amp_rpe']
        # go signal parameters
        self.thres = modelparams['thres']
        self.choice_mechanism = 'current'
        self.i_iti = params_ww['i_iti']
        self.w_iti = params_ww['w_iti']
        self.w_a_a = params_ww['w_a_a']
        self.w_b_b = params_ww['w_b_b']

        #line attractor
        self.la = line_attractor(modelparams, task)

        #wong&wang
        self.ww = WongWang(params_ww)

        #noise
        self.neu_indexes_pfc = np.array(range(modelparams['N_pfc']))

        self.amp_proj = modelparams['amp_ff']


    def _current_go_signal(self, l, t, r_a, r_b, curr_a, curr_b, noise_a, noise_b):
        '''current based go signal'''
        if t<=self.task.t_go:
            if r_a<self.thres and r_b<self.thres: 
                in_a = curr_a + noise_a + self.i_iti # this line might be problematic
                in_b = curr_b + noise_b + self.i_iti
        else:
            if (self.thres<=r_a or self.thres<=r_b): 
                l+=1
                if l==1:#firs time choice
                    choice = self.task.choice(r_a, r_b)
                    reward = self.task.foraging(choice)
                    self.task.average_reward(reward, choice)
                    if choice == 1:
                        rpe = self.task.rpe_a * self.la.null[0, :]
                    else:
                        rpe = self.task.rpe_b * self.la.null[1, :]
                    self.task.inputs['rpe_pfc'] = self.amp_rpe * rpe 
                    self.task.t_reward = t
                    l+=1
                if 1<l:
                    in_a = curr_a + noise_a + self.i_iti # this line might be problematic
                    in_b = curr_b + noise_b + self.i_iti
            else:
                if l==0:
                    in_a = curr_a + noise_a 
                    in_b = curr_b + noise_b 
                else:
                    in_a = curr_a + noise_a + self.i_iti # this line might be problematic
                    in_b = curr_b + noise_b + self.i_iti
        return in_a, in_b, l
    
    def _recurrent_weights_go_signal(self, l, t, r_a, r_b):
        '''synaptic based go signal'''
        if t<=self.task.t_go:
            if r_a<self.thres and r_b<self.thres: 
                self.ww.w_a_a = self.w_iti
                self.ww.w_b_b = self.w_iti
        else:
            if (self.thres<=r_a or self.thres<=r_b): 
                l+=1
                if l==1:#firs time choice
                    choice = self.task.choice(r_a, r_b)
                    reward = self.task.foraging(choice)
                    self.task.average_reward(reward, choice)
                    if choice == 1:
                        rpe = self.task.rpe_a * self.la.null[0, :]
                    else:
                        rpe = self.task.rpe_b * self.la.null[1, :]
                    print('rpe_a', self.task.rpe_a, 'rpe_b', self.task.rpe_b)
                    self.task.inputs['rpe_pfc'] = self.amp_rpe * self.la.null[0, :]#* rpe HEREEEE THIS CN LEAD TO PROBLEMS
                    self.task.t_reward = t
                    l+=1
                if 1<l:
                    self.ww.w_a_a = self.w_iti
                    self.ww.w_b_b = self.w_iti
            else:
                if l==0:
                    self.ww.w_a_a = self.w_a_a
                    self.ww.w_b_b = self.w_b_b
                else:
                    self.ww.w_a_a = self.w_iti
                    self.ww.w_b_b = self.w_iti
        return l
    
    # dynamics 
    def dynamics(self, u_init):
        ''' simulating the dynamics of neuronal network '''

        #Line attractor
        un_pfc = u_init['pfc0']
        xn_pfc = u_init['pfc_noise0']
        rn_pfc =  self.la.tf_sig(un_pfc)
        overlap_pfc = self.la._overlaps(rn_pfc)

        #Wong-wang
        s_a = u_init['s0_a']
        s_b = u_init['s0_b']
        noise_a = 0 #initial condition of the noise
        noise_b = 0
        curr_a = self.amp_proj * overlap_pfc[0]
        curr_b = self.amp_proj * overlap_pfc[1]
        in_a = curr_a + noise_a 
        in_b = curr_b + noise_b 
        
        #appending dynamics
        rates_pfc = [] #neurons dynammics LA
        ovs_pfc = [] # overlap LA
        rates_ww = [] #rates WW

        t=0
        l=0 # index after you hit the threshold
        while t<=self.period:
            #Line attractor
            un_pfc = un_pfc + self.dt * self.la.field(un_pfc, xn_pfc, t) 
            xn_pfc = xn_pfc + self.dt * self.la.noise(xn_pfc)
            rn_pfc =  self.la.tf_sig(un_pfc)
            overlap_pfc = self.la._overlaps(rn_pfc)
            rates_pfc.append(rn_pfc[self.neu_indexes_pfc])
            ovs_pfc.append(overlap_pfc)

            #Wong&Wang
            f_a, f_b, r_a, r_b = self.ww.field_one_area(s_a, s_b, in_a, in_b)
            s_a = s_a + self.ww.dt * f_a
            s_b = s_b + self.ww.dt * f_b
            f_noise_a, f_noise_b = self.ww.field_colored_noise(noise_a, noise_b)
            noise_a = noise_a + f_noise_a
            noise_b = noise_b + f_noise_b
            rates_ww.append([r_a, r_b])

            if self.choice_mechanism == 'current':
                # update in_a and in_b internally
                in_a, in_b, l = self._current_go_signal(l, t, r_a, r_b, curr_a, curr_b, noise_a, noise_b)
                curr_a = self.amp_proj * overlap_pfc[0]
                curr_b = self.amp_proj * overlap_pfc[1]
            elif self.choice_mechanism == 'recurrent':
                l = self._recurrent_weights_go_signal(l, t, r_a, r_b)
                in_a = curr_a + noise_a 
                in_b = curr_b + noise_b 
                curr_a = self.amp_proj * overlap_pfc[0]
                curr_b = self.amp_proj * overlap_pfc[1]
            t += self.dt

        #rest rpe after trial
        self.task.rpe_a = 0 
        self.task.rpe_b = 0 
        dynamics = dict(
                #Line attractor variables
                rates_pfc = rates_pfc,
                un_pfc = un_pfc,
                xn_pfc = xn_pfc,
                overlaps_pfc = ovs_pfc,
                
                #wong-wang variables
                sn_a = s_a,
                sn_b = s_b,
                rates_ww = rates_ww,

                #task variables
                reaction_time = self.task.t_reward - self.task.t_go 
                )
        for key in dynamics:
            dynamics[key] = np.array(dynamics[key])
        print('reaction time', dynamics['reaction_time'])
        return dynamics

