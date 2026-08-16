import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy as sp
from utils import signalprocessing as sig
import matplotlib.pyplot as plt
from utils import models
import time

def reformattingKspaceForSpark(inputKspace, kspaceOriginal, acsregionX, acsregionY, acsx, acsy, normalizationflag):
    [E, C, _, _] = inputKspace.shape
    kspaceAcsCrop = kspaceOriginal[:, :, acsregionX[0]:acsregionX[acsx - 1] + 1, acsregionY[0]:acsregionY[acsy - 1] + 1]
    kspaceAcsGrappa = inputKspace[:, :, acsregionX[0]:acsregionX[acsx - 1] + 1, acsregionY[0]:acsregionY[acsy - 1] + 1]
    kspaceAcsDifference = kspaceAcsCrop - kspaceAcsGrappa

    acs_difference_real = np.real(kspaceAcsDifference)
    acs_difference_imag = np.imag(kspaceAcsDifference)

    # Adding the batch dimension
    kspace_grappa = np.copy(inputKspace)
    kspace_grappa_real = np.real(kspace_grappa)
    kspace_grappa_imag = np.imag(kspace_grappa)
    kspace_grappa_split = np.concatenate((kspace_grappa_real, kspace_grappa_imag), axis=1)

    chan_scale_factors_real = np.zeros((E, C), dtype='float')
    chan_scale_factors_imag = np.zeros((E, C), dtype='float')

    for e in range(E):
        if (normalizationflag):
            scale_factor_input = 1 / np.amax(np.abs(kspace_grappa_split[e, :, :, :]))
            kspace_grappa_split[e, :, :, :] *= scale_factor_input

        for c in range(C):
            if (normalizationflag):
                scale_factor_real = 1 / np.amax(np.abs(acs_difference_real[e, c, :, :])) ##计算acs残差图像的实部与虚部归一化因子
                scale_factor_imag = 1 / np.amax(np.abs(acs_difference_imag[e, c, :, :]))
            else:
                scale_factor_real = 1
                scale_factor_imag = 1

            chan_scale_factors_real[e, c] = scale_factor_real
            chan_scale_factors_imag[e, c] = scale_factor_imag

            acs_difference_real[e, c, :, :] *= scale_factor_real
            acs_difference_imag[e, c, :, :] *= scale_factor_imag

    acs_difference_real = np.expand_dims(acs_difference_real, axis=2)
    acs_difference_real = np.expand_dims(acs_difference_real, axis=2)
    acs_difference_imag = np.expand_dims(acs_difference_imag, axis=2)
    acs_difference_imag = np.expand_dims(acs_difference_imag, axis=2)

    kspace_grappa_split = torch.from_numpy(kspace_grappa_split)
    kspace_grappa_split = kspace_grappa_split.to(device, dtype=torch.float)

    acs_difference_real = torch.from_numpy(acs_difference_real)
    acs_difference_real = acs_difference_real.to(device, dtype=torch.float)

    acs_difference_imag = torch.from_numpy(acs_difference_imag)
    acs_difference_imag = acs_difference_imag.to(device, dtype=torch.float)

    return kspace_grappa_split, acs_difference_real, acs_difference_imag, chan_scale_factors_real, chan_scale_factors_imag



def trainingSparkNetwork(kspaceGrappaSplit, acsDifferenceReal, acsDifferenceImag, acsx, acsy, learningRate, iterations):
    [E, C, _, _, _, _] = acsDifferenceReal.shape

    real_model = models.SPARK_Netv2_Residual_one_model_plus0(coils=C, kernelsize=3, acsx=acsx, acsy=acsy)
    real_model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(real_model.parameters(), lr=learningRate)
    real_losses = []
    print('Training Real Model')
    for epoch in range(iterations):
        optimizer.zero_grad()
        _, loss_out = real_model(kspaceGrappaSplit)
        loss = criterion(loss_out, torch.unsqueeze(torch.squeeze(acsDifferenceReal),axis = 0))
        loss.backward()
        optimizer.step()
        running_loss = loss.item()
        real_losses.append(running_loss)
        if epoch % 100 == 0:
            print('Epoch [{}/{}], Loss: {:.10f}'.format(epoch, iterations, running_loss))
    print('Training Complete, loss = %.10f' % (running_loss))

    imag_model = models.SPARK_Netv2_Residual_one_model_plus0(coils=C, kernelsize=3, acsx=acsx, acsy=acsy)
    imag_model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(imag_model.parameters(), lr=learningRate)
    imag_losses = []
    print('Training Imaginary Model')
    for epoch in range(iterations):
        optimizer.zero_grad()
        _, loss_out = imag_model(kspaceGrappaSplit)
        loss = criterion(loss_out, torch.unsqueeze(torch.squeeze(acsDifferenceImag),axis = 0))
        loss.backward()
        optimizer.step()
        running_loss = loss.item()
        imag_losses.append(running_loss)
        if epoch % 100 == 0:
            print('Epoch [{}/{}], Loss: {:.10f}'.format(epoch, iterations, running_loss))
    print('Training Complete, loss = %.10f' % (running_loss))
    return real_model, imag_model, real_losses, imag_losses

def plot_losses(real_losses, imag_losses):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(real_losses, label='Real Part Loss')
    plt.title('Real Part Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(imag_losses, label='Imaginary Part Loss')
    plt.title('Imaginary Part Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

def applySparkCorrection(kspaceToCorrect, kspaceGrappaSplit, real_model, imag_model, chanScaleFactorReal, chanScaleFactorImag):
    correctionr = real_model(kspaceGrappaSplit)[0].cpu().detach().numpy()
    correctioni = imag_model(kspaceGrappaSplit)[0].cpu().detach().numpy()
    corrected = correctionr / chanScaleFactorReal[:, :, np.newaxis, np.newaxis] + 1j * correctioni / chanScaleFactorImag[:, :, np.newaxis, np.newaxis] + kspaceToCorrect
    return corrected

time_start = time.time()

kspace = np.expand_dims(np.transpose(sp.io.loadmat('grappa_recons/kspace_full.mat')['kspace'], axes=(2, 0, 1)), axis=0)
[E, C, M, N] = kspace.shape
kspace0 = np.copy(kspace)
weight = np.repeat(sp.io.loadmat('data/weight/weight2_320_320.mat')['weight'][:,:,np.newaxis],C,axis=-1) #定
weight = np.expand_dims(weight, axis=0)
weight = weight.transpose(0,3,1,2)
kspace = np.multiply(kspace, weight)
normalizationflag = 0
normalizeAll = 0
Rx = 1
acsx = M
all_accelerations = [4]
all_acs_sizes = [20]
all_iterations = [500]
all_learning_rates = [.04]
A = len(all_accelerations)
S = len(all_acs_sizes)
I = len(all_iterations)
R = len(all_learning_rates)
all_kspaces_spark = np.zeros((I, R, A, S, E, C, M, N), dtype=complex)
all_kspaces_grappa = np.zeros((I, R, A, S, E, C, M, N), dtype=complex)

ctr = 1
for aa in range(A):
    for ss in range(S):
        for ii in range(I):
            for rr in range(R):
                Ry = all_accelerations[aa]
                acsy = all_acs_sizes[ss]
                iterations = all_iterations[ii]
                learningRate = all_learning_rates[rr]
                print('Recon %d/%d || R %d || acsy %d || s %.5f || it %d' % \
                      (ctr, A * S * I * R, Ry, acsy, learningRate, iterations))
                ctr += 1
                kspaceGrappa = np.expand_dims(np.transpose(sp.io.loadmat( \
                'grappa_recons/kspace_grappa_Rx1Ry%dacsx%dacsy%d.mat' % \
                (Ry, acsx, acsy))['kspace_grappa'], axes=(2, 0, 1)), axis=0)
                kspaceGrappa = np.multiply(kspaceGrappa, weight)

                acsregionX = np.arange(M // 2 - acsx // 2, M // 2 + acsx // 2)
                acsregionY = np.arange(N // 2 - acsy // 2, N // 2 + acsy // 2)
                kspaceAcsZerofilled = np.zeros((E, C, M, N), dtype=complex)
                kspaceAcsZerofilled[:, :, acsregionX[0]:acsregionX[acsx - 1] + 1, \
                acsregionY[0]:acsregionY[acsy - 1] + 1] = kspace[:, :, acsregionX[0]:acsregionX[acsx - 1] + 1, \
                                                          acsregionY[0]:acsregionY[acsy - 1] + 1]


                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                [kspace_grappa_split, acs_difference_real, acs_difference_imag, chan_scale_factors_real, \
                 chan_scale_factors_imag] = reformattingKspaceForSpark(kspaceGrappa, kspaceAcsZerofilled, acsregionX, \
                                                                      acsregionY, acsx, acsy, normalizationflag)


                real_model, imag_model, real_losses, imag_losses = trainingSparkNetwork(kspace_grappa_split, \
                                                                                      acs_difference_real, \
                                                                                      acs_difference_imag, acsregionX, \
                                                                                      acsregionY, learningRate, iterations)

                plot_losses(real_losses, imag_losses)
                kspaceCorrected = np.zeros((E, C, M, N), dtype=complex)

                kspaceToCorrect = kspaceGrappa
                kspaceGrappaSplit = kspace_grappa_split

                acsx = M
                kspaceGrappaSplit = sig.real_to_complex_torch(kspaceGrappaSplit, 1)
                kspaceGrappaSplit = kspaceGrappaSplit.unsqueeze(0)
                kspace1 = np.copy(kspace)
                kspace1 = torch.from_numpy(kspace1)
                kspace1 = kspace1.to(device, dtype=torch.float)
                kspace1 = kspace1.unsqueeze(0)
                acsregionX = np.arange(M // 2 - acsx // 2, M // 2 + acsx // 2)
                acsregionY = np.arange(N // 2 - acsy // 2, N // 2 + acsy // 2)
                kspaceGrappaSplit[:, :, :, acsregionX[0]:acsregionX[acsx - 1] + 1,
                acsregionY[0]:acsregionY[acsy - 1] + 1] = kspace1[:, :, :, acsregionX[0]:acsregionX[acsx - 1] + 1,
                                                          acsregionY[0]:acsregionY[acsy - 1] + 1]
                kspaceGrappaSplit = sig.complex_to_real_torch(kspaceGrappaSplit, 2)
                kspaceGrappaSplit = kspaceGrappaSplit.squeeze(0)

                kspaceCorrected = applySparkCorrection(kspaceToCorrect, kspaceGrappaSplit, real_model, imag_model, \
                                                        chan_scale_factors_real, \
                                                        chan_scale_factors_imag)
                kspaceCorrected = np.multiply(kspaceCorrected, 1. / weight)
                kspace = np.copy(kspace0)
                kspaceGrappa = np.multiply(kspaceGrappa, 1. / weight)

                kspaceCorrectedReplaced = np.copy(kspaceCorrected)
                kspaceCorrectedReplaced[:, :, acsregionX[0]:acsregionX[acsx - 1], acsregionY[0]:acsregionY[acsy - 1]] = \
                    kspace[:, :, acsregionX[0]:acsregionX[acsx - 1], acsregionY[0]:acsregionY[acsy - 1]]
                kspaceGrappaReplaced = np.copy(kspaceGrappa)
                kspaceGrappaReplaced[:, :, acsregionX[0]:acsregionX[acsx - 1], acsregionY[0]:acsregionY[acsy - 1]] = \
                    kspace[:, :, acsregionX[0]:acsregionX[acsx - 1], acsregionY[0]:acsregionY[acsy - 1]]

                all_kspaces_spark[ii, rr, aa, ss, :, :, :, :] = kspaceCorrectedReplaced
                all_kspaces_grappa[ii, rr, aa, ss, :, :, :, :] = kspaceGrappaReplaced

time_end = time.time()
time_elapsed = time_end - time_start
print('t:',time_elapsed,'s')
results = {
           'kspace': np.squeeze(kspace),
           'grappa_kspace': np.squeeze(all_kspaces_grappa),
           'res_spark_w2_kspace': np.squeeze(all_kspaces_spark),
           'time_w2': time_elapsed,
           }
sp.io.savemat('results/f_res_spark_ablation_w2_one.mat', results, oned_as='row')

