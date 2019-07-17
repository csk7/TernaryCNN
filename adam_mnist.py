import os
import time
import math
import torch
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import numpy as np
from torch.distributions import Bernoulli
from numpy import random


import conv1_bias
import conv1_weight
import conv2_bias
import conv2_weight
import fc1_bias
import fc1_weight
import fc2_bias
import fc2_weight


transform=transforms.Compose([transforms.ToTensor()])

torch.set_printoptions(precision=10)

def init_weights(m):
    #print(m)
    torch.cuda.manual_seed(random.randint(1,2147462579))

    if type(m) == (nn.Conv2d):
        nn.init.uniform(m.weight,-1.5,1.5)
        
    if type(m) == (nn.Linear):
        nn.init.uniform(m.weight,-1.5,1.5)
        


class RoundNoGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()
    @staticmethod
    def backward(ctx, g):
        return g 

def hard_sigmoid(x):
    return torch.clamp((x+1.)/2,0,1)

def discrete_neuron_3states(x):
    return RoundNoGradient.apply(hard_sigmoid(2*(x-1))+hard_sigmoid(2*(x+1))-1)

def weight_tune(weights,l_limit,r_limit):
    
    state_index = torch.round((weights - l_limit)/(r_limit - l_limit)*pow(2,N))
    weights = state_index/pow(2,N)*(r_limit-l_limit) + l_limit
    return weights
    
class Adam(optim.Adam):
    def __init__(self,params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,weight_decay=0):
        super(Adam,self).__init__(params,lr=lr, betas=betas, eps=eps,weight_decay=weight_decay)

    def false_step(self,cur_lr):

        W_delta = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = cur_lr * math.sqrt(bias_correction2) / bias_correction1
                
                W_delta.append(step_size * torch.div(exp_avg, denom))
        return W_delta    

def hinge_loss(outputs,labels):
        lb2 = torch.LongTensor(my_batch,1).cuda()
        lb2[:,0] = labels
        y_onehot = (torch.FloatTensor(my_batch, 10)).cuda()
        y_onehot.zero_()
        y_onehot.scatter_(1, lb2, 1)

        y_final = Variable(y_onehot.cuda())
        y_f = 2*y_final - 1

        l1 = outputs * y_f
        l2 = 1. - l1
        l_zero = Variable(torch.zeros(my_batch,10).cuda())
        l3 = torch.max(l_zero,l2)
        l4 = torch.mul(l3,l3)
        loss = l4.mean()
        return loss 



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.pool2 = nn.MaxPool2d(2,2)
        


        self.fc1 = nn.Linear( 64* 4 * 4, 50)
        self.fc2 = nn.Linear(50, 10)
    
        
    
    def forward(self, x):
        x = discrete_neuron_3states(self.pool1(self.conv1(x)))      
        x = discrete_neuron_3states(self.pool2(self.conv2(x)))
     
        
        x = x.view(-1,64 * 4 * 4)
        x = discrete_neuron_3states(self.fc1(x))
        x = self.fc2(x)

        return x
                    

def discrete_grads(net,delta,iters):   

    for (name,i),j,kc in zip(net.named_parameters(),delta,range(0,len(delta))):
        if 'bias' not in name:
            L = 2*H / pow(2,N)
            delta_W2 = -1 * j
            delta_W2_direction = torch.sign(delta_W2)
            dis2=torch.abs(delta_W2)
            k2=delta_W2_direction*torch.floor(dis2/L)
            v2=delta_W2-k2*L
            Prob2 = torch.abs(v2/L)
            Prob2 = torch.tanh(th*Prob2)
            torch.cuda.manual_seed(random.randint(1,2147462579))
            Gate2_class = Bernoulli(Prob2)
            Gate2 = Gate2_class.sample()
            delta_W2_new=(k2+delta_W2_direction*Gate2)*L
            updates_param2 = torch.clamp(i.data + delta_W2_new,-H,H)
            updates_param2 = weight_tune(updates_param2,-H,H)
            i.data = updates_param2

        else:
            i.data -= j 

            
def random_discrete_grads(net,delta,iters):
 

    for (name,i),j,kc in zip(net.named_parameters(),delta,range(0,len(delta))):
        if 'weight' in name:
            L = 2*H / pow(2,N)
            torch.cuda.manual_seed(random.randint(1,2147462579))
            a = torch.rand(1).cuda()
            c = torch.zeros(1).cuda()
            if a[0] < 0.8:
                c[0] = 1
            else:
                c[0] = 0

            b = torch.rand(1).cuda()
            #xy = b * 2
            #print(xy.type())
            state_rand = torch.round(b*2.)*L -H
            
            delta_W2 = c[0]*(state_rand - i.data)
            
            delta_W2_direction = torch.sign(delta_W2)

            dis2=torch.abs(delta_W2)
            k2=delta_W2_direction*torch.floor(dis2/L)
            v2=delta_W2-k2*L
            Prob2 = torch.abs(v2/L)

            Prob2 = torch.tanh(th*Prob2)
            torch.cuda.manual_seed(random.randint(1,2147462579))
            Gate2_class = Bernoulli(Prob2)
            Gate2 = Gate2_class.sample()
            
            delta_W2_new=(k2+delta_W2_direction*Gate2)*L
            updates_param2 = torch.clamp(i.data + delta_W2_new,-H,H)
            updates_param2 = weight_tune(updates_param2,-H,H)
            i.data = updates_param2
            
            
        if 'bias' in name:
            i.data -= j      

def train():

    net = Net()       
    net.cuda()
    net.apply(init_weights)
    optimizer = Adam(net.parameters(), lr=LR_start) 
    best_epoch =1
    update_type = 10
    best_acc = 0.0
    t1 = time.clock();
    for epoch in range(num_epochs): 
        running_loss = 0.0
        if epoch == 0:
            current_lr = 0;
        elif epoch == 1:
            current_lr = LR_start*0.005;
        else:
            current_lr = current_lr * LR_decay


        print(current_lr)


        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs1 = 2*inputs -1 
            inputs, labels = Variable(inputs1.cuda()), labels.cuda()
            net.train()
            net.zero_grad()
            outputs = net(inputs.cuda())
            loss = hinge_loss(outputs,labels)
            loss.backward()
            delta = optimizer.false_step(current_lr)
            discrete_grads(net,delta,epoch)
            running_loss += loss.data[0]


        t2 = time.clock()    
        print('Epoch : %d Time : %.3f sec and loss: %.3f' % (epoch + 1, (t2- t1), running_loss / i))
        accuracy_load = test_acc(net)
        print('test_acc: %.3f %%' % accuracy_load)


        if epoch == 15:
            for name,i in net.named_parameters():
                if 'bias' in name:
            	    if 'conv1' in name:
                        print(name)
                        conv1_bias.single_line_write(i)
                        print(i.size())

            	    elif 'conv2' in name:
                        print(name)
                        conv2_bias.single_line_write(i)
                        print(i.size())

            	    elif 'fc1' in name:
                        print(name)
                        fc1_bias.single_line_write(i)
                        print(i.size())

            	    elif 'fc2' in name:
                        print(name)
                        fc2_bias.single_line_write(i)
                        print(i.size())

                else:
            	    if 'conv1' in name:
                        print(name)
                        conv1_weight.write_4d(i)
                        print(i.size())

            	    elif 'conv2' in name:
                        print(name)
                        conv2_weight.write_4d(i)
                        print(i.size())

            	    elif 'fc1' in name:
                        print(name)
                        fc1_weight.write_2d(i)
                        print(i.size())

            	    elif 'fc2' in name:
                        print(name)
                        fc2_weight.write_2d(i)
                        print(i.size())                                        

    print('Finished Training')

def train_acc(net):
    correct = 0
    total = 0
    for data in trainloader:
        images, labels = data
        outputs = net(Variable(images.cuda()))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.cuda()).sum()

    print('Accuracy on the 60000 train images: %.3f %%' % (100 * correct / total))

def test_acc(net):
    net.eval()
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        images1 = 2*images - 1
        outputs = net(Variable(images1.cuda()))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.cuda()).sum()
    print('Accuracy on the 10000 test images: %.3f %%' % (100 * correct / total))
    return (100 * correct / total)


num_epochs = 4000
LR_start = 0.1
LR_fin = 0.0000001
LR_decay = (LR_fin/LR_start)**(1./(num_epochs)) 

H = 1.
N = 1.
th = 3.

my_batch = 48
#print('hello')
#print(os.path.dirname(os.path.abspath(__file__)))
#print('bye')
path = os.path.join(os.path.expanduser('~'), 'Thesis', 'Hiver18','GXNOR','V1','MNIST', 'state1.pth')
trainset = torchvision.datasets.MNIST(root='./data', train=True,download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=my_batch,shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False,download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=my_batch,shuffle=False, num_workers=2)



train()
test_acc()



