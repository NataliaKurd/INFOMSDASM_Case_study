import numpy
import matplotlib.pyplot as plt

labelsize = 7

##############
# households #
##############

# default propensity
a = 0.0

# twice the height of the logistic curve for households
betaH = 8.0

# 1/range of logistic curve for households
gammaH = 0.8

# slope of curve at propensity = 0
resultingSlopeAtZero = (gammaH*betaH)/4.0

#################################
# food environment of household #
#################################
# proportion
# should be between <0,1> otherwise propensity goes to inf or -inf
# this is the 'magnitude' of the food environment effect
proportionOne = 0.7
#proportionOne = 0.000001

# twice the height of the logistic curve for food environment effect
betaS = proportionOne * betaH

# should be in <0,>, for > 1, three equilibria (for 1 household per outlet)
proportionTwo = 4.0

# 1/range of logistic curve for food environment effect
gammaS = ((4*resultingSlopeAtZero)/betaS) * proportionTwo

# differential equation internal effects
def diffEqTermOne(x, a, betaH, gammaH):
    return -((betaH / (1.0 + numpy.exp(-gammaH * (x - a)))) - (betaH / 2.0))

# differential equation food outlet effects
def diffEqTermTwo(y, a, betaS, gammaS):
    return ((betaS / (1.0 + numpy.exp(-gammaS * (y - a)))) - (betaS / 2.0))

# plotting the differential equation
x = numpy.arange(-10,10,0.01)
y = x
termOne = diffEqTermOne(x, a, betaH, gammaH)
termTwo = diffEqTermTwo(y, a, betaS, gammaS)

f = plt.figure()
f, axs = plt.subplots(3,3, gridspec_kw={'hspace': 0.3, 'wspace': 0.0}, sharey='row')

axs[0,0].plot(x,termOne, label='hh')
axs[0,1].plot(y,y/10000000, label='fo')
axs[0,2].plot(x,termOne, label='tot')

propensityStores=0.2
rate = diffEqTermTwo((y/10000)+ propensityStores, a, betaS, gammaS)
axs[1,0].plot(x,termOne, label='hh')
axs[1,1].plot(y,rate, label='fo')
axs[1,2].plot(x,termOne + rate, label='tot')

axs[2,0].plot(x,termOne, label='hh')
axs[2,1].plot(y,termTwo, label='fo')
axs[2,2].plot(x,termOne + termTwo, label='tot')

for ax in axs:
  for a in ax:
    a.set_xlim((-4,4))
    a.set_ylim((-5,5))
    a.hlines([0],-5,5,linestyle='-', linewidth=0.5)
    a.tick_params(axis='x', labelsize=labelsize )
    a.set_xticks((-3,-2,-1,0,1,2,3))
    a.title.set_size(labelsize * 1.5)
    #a.legend(fontsize=labelsize, ncol=3, loc=8)

axs[2,0].set_xlabel('propensity household', fontsize=labelsize)
axs[2,1].set_xlabel('propensity household', fontsize=labelsize)
axs[2,2].set_xlabel('propensity household', fontsize=labelsize)

axs[0,0].set_ylabel('rate of change in\n propensity (year-1)', fontsize=labelsize)
axs[1,0].set_ylabel('rate of change in\n propensity (year-1)', fontsize=labelsize)
axs[2,0].set_ylabel('rate of change in\n propensity (year-1)', fontsize=labelsize)

axs[0,0].title.set_text('household effect')
axs[0,1].title.set_text('food store effect')
axs[0,2].title.set_text('total effect')

axs[0,1].text(-3.4, 3.5, 'no effect', size=labelsize)
axs[1,1].text(-3.4, 3.5, 'food st. prop. = 0.2', size=labelsize)
axs[2,1].text(-3.4, 3.5, 'food st. prop. = househ. prop.', size=labelsize)

f.savefig("output/equations.pdf")


f = plt.figure()

# plot the curve
plt.plot(y,termTwo, label='fo')

# plot the line
#rate = diffEqTermTwo((y/10000)+ propensityStores, a, betaS, gammaS)
xLine = numpy.arange(-4,propensityStores,0.01)
yLine = (xLine / 10000) + numpy.mean(rate)
plt.plot(xLine,yLine, linestyle='dotted', color = 'orange')
plt.xlabel("propensity of food stores, y")
plt.ylabel('contribution to rate of change in\n propensity of households (year-1)')
plt.hlines([0],-5,5,linestyle='-', linewidth=0.5)
plt.vlines([propensityStores],0,numpy.mean(rate),linestyle='dotted', color = 'orange')

plt.xlim(-4,4)
plt.ylim(-5,5)

f.savefig("output/equation_food_store_effect.pdf")
