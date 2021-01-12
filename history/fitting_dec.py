import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from array import array
import sys
import xlrd
from openpyxl import Workbook
from scipy.integrate import odeint
import csv

data_path = "D:/py_projects/covid/data/"
# xdata = range(0,len(ydata),1)
# def sigmoid(x, x0, k):
#  y = 1 / (1+ np.exp(-k*(x-x0)))
#   return y
# popt, pcov = curve_fit(sigmoid, xdata, ydata)
# yOut = 1 / (1+ np.exp(-popt[1]*(xdata-popt[0])))
df = pd.read_csv(data_path + "Covid_cases.csv", usecols=['Entity', 'Code', 'cases', 'Days_30'])
# df = df[~(df == 0).any(axis=1)]
df = df[df['Days_30'] >= 0]
df = df.dropna(subset=['Code'])
# df=df.dropna(subset=['Date_30'])
# print(df['Entity'][0:5])
# df1 = pd.read_csv("incidence.csv")
df = df.loc[(df['Entity'] != ('Bahamas'))]
df = df.loc[(df['Entity'] != ('Benin'))]
df = df.loc[(df['Entity'] != ('Botswana'))]
df = df.loc[(df['Entity'] != ('Gambia'))]
df = df.loc[(df['Entity'] != ('Kazakhstan'))]
df = df.loc[(df['Entity'] != ('Kyrgyzstan'))]
df = df.loc[(df['Entity'] != ('Latvia'))]
df = df.loc[(df['Entity'] != ('Lebanon'))]
df = df.loc[(df['Entity'] != ('Sao Tome and Principe'))]
df = df.loc[(df['Entity'] != ('Uganda'))]
df = df.loc[(df['Entity'] != ('Andorra'))]
df = df.loc[(df['Entity'] != ('Angola'))]
df = df.loc[(df['Entity'] != ('Came Verde'))]
df = df.loc[(df['Entity'] != ('Nicaragua'))]
df = df.loc[(df['Entity'] != ('Sri Lanka'))]
df = df.loc[(df['Entity'] != ('Tunisia'))]
df = df.loc[(df['Entity'] != ('Western Sahara'))]
df = df.loc[(df['Entity'] != ('Yemen'))]
df = df.loc[(df['Entity'] != ('Djibouti'))]
df = df.loc[(df['Entity'] != ('Guinea Bissau'))]
df = df.loc[(df['Entity'] != ('Cyprus'))]
df = df.loc[(df['Entity'] != ('Maldives'))]
df = df.loc[(df['Entity'] != ('Sierra Leone'))]


# df=df.loc[(df['Entity'] !=('Kazakhstan'))]
# df1=df1.loc[(df1['Entity'] !=('Faeroe Islands'))]
# df1=df1.loc[(df1['Entity'] !=('Kosovo'))]
# df1=df1.loc[(df1['Entity'] !=('Kyrgyzstan'))]
# df1=df1.loc[(df1['Entity'] !=('Latvia'))]
# df1=df1.loc[(df1['Entity'] !=('Lebanon'))]
# df1=df1.loc[(df1['Entity'] !=('Lithuania'))]
# df1=df1.loc[(df1['Entity'] !=('Tunisia'))]
# df1=df1.loc[(df1['Entity'] !=('Niger'))]
# df1=df1.loc[(df1['Entity'] !=('Bosnia and Herzegovina'))]
# 'Niger', 'Bosnia and Herzegovina'
# Faeroe Islands', 'Kosovo', 'Kyrgyzstan', 'Latvia', 'Lebanon', 'Lithuania', 'Tunisia
# df=pd.merge(df1, df, on='Entity')
# df=df.loc[(df['Entity'] =='Afghanistan')|(df['Entity'] =='Albania')|(df['Entity'] =='Algeria')|(df['Entity'] =='Andorra')|(df['Entity'] =='Argentina')|(df['Entity'] =='Armenia')|(df['Entity'] =='Australia')|(df['Entity'] =='Austria')|(df['Entity'] =='Azerbaijan')|(df['Entity'] == 'Bahrain')|(df['Entity'] == 'Belarus')|(df['Entity'] =='Belgium')|(df['Entity'] =='Bolivia')|(df['Entity'] =='Bosnia and Herzegovina')|(df['Entity'] =='Brazil')|(df['Entity'] =='Brunei')|(df['Entity'] =='Bulgaria')|(df['Entity'] =='Burkina Faso')|(df['Entity'] =='Cambodia')|(df['Entity'] =='Cameroon')|(df['Entity'] =='Canada')|(df['Entity'] =='Chile')|(df['Entity'] == 'China')|(df['Entity'] == 'Colombia')|(df['Entity'] == 'Costa Rica')|(df['Entity'] == "Cote d'Ivoire")|(df['Entity'] == 'Croatia')|(df['Entity'] == 'Cuba')|(df['Entity'] == 'China')|(df['Entity'] == 'Cyprus')|(df['Entity'] =='Czech Republic')|(df['Entity'] == 'Denmark')|(df['Entity'] == 'Dominican Republic')|(df['Entity'] == 'Ecuador')|(df['Entity'] == 'Egypt')|(df['Entity'] =='Estonia')|(df['Entity'] =='Faeroe Islands')|(df['Entity']=='Finland')|(df['Entity']=='Georgia')|(df['Entity'] == 'France')| (df['Entity'] =='Germany')|(df['Entity'] =='Ghana')|(df['Entity'] =='Greece')|(df['Entity'] =='Honduras')|(df['Entity'] =='Hong Kong')|(df['Entity'] =='Hungary')|(df['Entity'] =='Iceland')|(df['Entity'] =='India')|(df['Entity'] =='Indonesia')|(df['Entity'] == 'Iran')|(df['Entity'] == 'Iraq')|(df['Entity'] =='Ireland')|(df['Entity'] =='Israel')|(df['Entity'] =='Italy')|(df['Entity'] == 'Japan')|(df['Entity'] == 'Jordan')|(df['Entity'] == 'Kazakhstan')|(df['Entity'] == 'Kosovo')|(df['Entity'] == 'Kuwait')|(df['Entity'] == 'Kyrgyzstan')|(df['Entity'] =='Latvia')|(df['Entity'] =='Lebanon')|(df['Entity'] =='Lithuania')|(df['Entity'] =='Luxembourg')|(df['Entity'] =='Malaysia')|(df['Entity'] =='Malta')|(df['Entity'] =='Mauritius')|(df['Entity'] =='Mexico')|(df['Entity'] =='Moldova')|(df['Entity'] =='Montenegro')|(df['Entity'] =='Morocco')|(df['Entity'] =='Netherlands')|(df['Entity'] =='New Zealand')|(df['Entity'] =='Nigeria')|(df['Entity'] =='Norway')|(df['Entity'] =='Oman')|(df['Entity'] =='Pakistan')|(df['Entity'] =='Palestine')|(df['Entity'] =='Panama')|(df['Entity'] =='Peru')|(df['Entity'] =='Philippines')|(df['Entity'] =='Poland')|(df['Entity'] =='Portugal')| (df['Entity'] =='Puerto Rico')|(df['Entity'] =='Qatar')|(df['Entity'] =='Romania')|(df['Entity'] =='Russia')|(df['Entity'] =='San Marino')|(df['Entity'] =='Saudi Arabia')|(df['Entity'] =='Senegal')|(df['Entity'] =='Serbia')|(df['Entity'] =='Singapore')|(df['Entity'] == 'South Korea')|(df['Entity'] =='Slovakia')|(df['Entity'] =='Slovenia')|(df['Entity'] =='South Africa')|(df['Entity'] == 'Spain')|(df['Entity'] == 'Sri Lanka')|(df['Entity'] == 'Sweden')|(df['Entity'] == 'Switzerland')|(df['Entity'] == 'Taiwan')|(df['Entity'] == 'Thailand')|(df['Entity'] == 'Tunisia')|(df['Entity'] =='Turkey')|(df['Entity'] =='Ukraine')|(df['Entity'] == 'United Kingdom')|(df['Entity'] == 'United States')|(df['Entity'] == 'Uruguay')|(df['Entity'] == 'Uzbekistan')|(df['Entity'] == 'Venezuela')|(df['Entity'] == 'Vietnam')]
# iset = ['china', 'korea']
def diff(f, x, r, k):
    didt = r * f * (1 - f / k)
    return didt


# df=df.drop(['Cameroon'], inplace = True)
result_df = df.drop_duplicates(subset=['Entity'], keep='first')
# writer=pd.ExcelWriter('july_data1.xlsx')
# result_df.to_excel(writer, sheet_name='JulyData_jude', index=False)
# writer.save()
# writer.close()
# sys.exit()

# result_df=result_df.drop(result_df['Entity']=='Cameroon', axis=0)
# , 'Faeroe Islands', 'Kosovo', 'Kyrgyzstan', 'Latvia', 'Lebanon', 'Lithuania', 'Tunisia'
countries = result_df['Entity'].values
# countries=['United States']
c1 = []
c2 = []
c3 = []
c4 = []
c5 = []
RR = []
temp = []
R0 = []
tot = []
# for i in iset:
#   c1.append(i)
# c2.append(1+2)
# me = ['US','canada']
# for i in me:
#    c1.append(i)
#   c2.append(10)
# df1 = pd.DataFrame({'C1': c1, 'C2': c2})
# print(df1)
# sys.exit()
# df=df.dropna()
# def logistic_model(x,b,c):
# return (ave*b)/(1+a*np.exp(-b*x))
#  return b*x*(1-x/c)
# countries=['Afghanistan', 'Albania', 'Algeria','Andorra','Argentina','Armenia', 'Australia', 'Austria', 'Azerbaijan', 'Bahrain','Belarus', 'Belgium','Bolivia', 'Bosnia and Herzegovina', 'Brazil', 'Brunei',
# 'Bulgaria', 'Burkina Faso','Cambodia', 'Cameroon', 'Canada', 'Chile','China','Colombia','Costa Rica',"Cote d'Ivoire", 'Croatia', 'Cuba', 'Cyprus','Czech Republic', 'Denmark','Dominican Republic', 'Ecuador', 'Egypt', 'Estonia', 'Faeroe Islands','Finland','France','Georgia', 'Germany','Ghana','Greece','Honduras','Hungary', 'Iceland', 'India', 'Indonesia', 'Iran','Iraq', 'Ireland','Israel','Italy', 'Japan','Jordan', 'Kazakhstan','South Korea','Kosovo', 'Kuwait', 'Kyrgyzstan','Latvia','Lebanon', 'Lithuania','Luxembourg','Malaysia', 'Malta','Mauritius','Mexico','Moldova','Montenegro', 'Morocco','Netherlands', 'New Zealand','Nigeria', 'Norway','Oman', 'Pakistan', 'Palestine', 'Panama', 'Peru', 'Philippines', 'Poland','Portugal', 'Puerto Rico', 'Qatar','Romania','Russia', 'San Marino', 'Saudi Arabia', 'Senegal', 'Serbia', 'Singapore', 'Slovakia','Slovenia','South Africa', 'Spain','Sri Lanka','Sweden', 'Switzerland', 'Taiwan', 'Thailand', 'Tunisia', 'Turkey','Ukraine','United Kingdom', 'United States', 'Uruguay', 'Uzbekistan', 'Venezuela', 'Vietnam']
# print(len(countries))
# sys.exit()
# scale1=[1.84, 2.24, 3.10, 2.50, 1.85, 2.25,  2.89,2.98,2.01, 2.15, 2.53 ,2.20, 2.03, 2.71, 3.58, 3.54, 2.15, 2.82, 2.14, 2.31,1.84, 2.08, 3.31, 2.24, 2.29, 2.01,  3.61, 2.61, 2.90, 3.91, 1.00, 3.09, 2.23, 2.06, 2.01, 2.60, 3.11, 1.99, 1.83]

# print((alphas*(1-thetasm))*sir[:,3]+ (alphay*(1-thetaym))*sir[:,2])
# print(xx)
# sys.exit()
D = {}
for item in countries:
    print(item)
    dftemp1 = df
    df = df.loc[(df['Entity'] == item)]
    xdata = df.Days_30.values
    # ydata=df.cases.values*scale1[countries.index(item)]
    ydata = df.cases.values
    # print(max(ydata))
    # print(ydata)
    xdata1 = xdata
    ydata1 = ydata

    if item == 'United States':
        ydata = ydata[:39 + 5]
        xdata = xdata[:39 + 5]
    elif item == 'Algeria':
        ydata = ydata[:15 + 5]
        xdata = xdata[:15 + 5]
    elif item == 'Australia':
        ydata = ydata[:18 + 5]
        xdata = xdata[:18 + 5]
    elif item == 'Bulgaria':
        ydata = ydata[:10 + 5]
        xdata = xdata[:10 + 5]
    elif item == 'Croatia':
        ydata = ydata[:8 + 5]
        xdata = xdata[:8 + 5]
    elif item == 'Canada':
        ydata = ydata[:50]
        xdata = xdata[:50]
    elif item == 'Israel':
        ydata = ydata[:20 + 5]
        xdata = xdata[:20 + 5]
    elif item == 'Japan':
        ydata = ydata[:40 + 5]
        xdata = xdata[:40 + 5]
    elif item == 'Macedonia':
        ydata = ydata[:10 + 5]
        xdata = xdata[:10 + 5]
    elif item == 'Morocco':
        ydata = ydata[:25 + 5]
        xdata = xdata[:25 + 5]
    elif item == 'Netherlands':
        ydata = ydata[:25 + 5]
        xdata = xdata[:25 + 5]
    elif item == 'Poland':
        ydata = ydata[:25 + 5]
        xdata = xdata[:25 + 5]
    elif item == 'Romania':
        ydata = ydata[:25 + 5]
        xdata = xdata[:25 + 5]
    elif item == 'Senegal':
        ydata = ydata[:20 + 5]
        xdata = xdata[:20 + 5]
    elif item == 'Serbia':
        ydata = ydata[:25 + 5]
        xdata = xdata[:25 + 5]
    elif item == 'Somalia':
        ydata = ydata[:25 + 5]
        xdata = xdata[:25 + 5]
    elif item == 'Sweden':
        ydata = ydata[:45 + 5]
        xdata = xdata[:45 + 5]
    elif item == 'Ukraine':
        ydata = ydata[:30 + 5]
        xdata = xdata[:30 + 5]
    else:
        ydata = ydata
        xdata = xdata
    ind = np.argmax(ydata)
    # print(ind)
    # print(ydata)
    # print(ydata[0:2])
    # print(ind)
    # print(ind+1)
    # print(ydata[15])
    ydata = ydata[:ind + 1]
    # print(ydata)
    # sys.exit()
    # print(ydata[:ind])
    # print(ydata.index(min(ydata)))
    # print(max(ydata))
    # sys.exit()

    # ydata=ydata[0:.index(max(a))]
    # print(min(ydata))
    # sys.exit()
    # print(xdata)
    # sys.exit()

    # print

    xdata = xdata[0:ind + 1]
    x = xdata

    # print(xdata)
    # print(ydata)
    # sys.exit()
    # print(x)
    # print(ydata)
    # print(ydata)
    # sys.exit()
    # inf.append(ydata[0])
    # I=ydata[0]

    D['Days_30' + str(item)] = xdata
    D['cases' + str(item)] = ydata
    # xx=np.linspace(0, len(ydata)-1)
    # f0=ydata[0]
    inf = []


    def curvefit_model(x, a, r, k):
        f0 = a
        f = odeint(diff, f0, x, args=(r, k))
        inff = np.zeros(len(f[:, 0]))
        inf.append(f[:, 0][0])
        inff[0] = ydata[0]
        # ydata[0]
        # inff[0]=f[:,0][1]-f[:,0][1]

        for ii in range(1, len(f[:, 0])):
            # inf.append(f[:,0][ii]-f[:,0][ii-1])
            inff[ii] = f[:, 0][ii] - f[:, 0][ii - 1]
        # print(inff)
        # print(f[:,0])
        # print(len(inf))
        # print(len(f[:,0]))
        # sys.exit()
        # len(inf)
        # sys.exit()
        # model=np.interp(x, xx,f)
        return inff


    # print(I)
    # print(ydata)
    # print(xdata)
    # sys.exit()
    # if item==['Gabon' 'Benin'] :
    p0 = [5, 1, 10000]
    # else:
    # p0=[2,1,2000]
    popt1, pcov1 = curve_fit(curvefit_model, x, ydata, p0)
    a, r, k = popt1
    c1.append(item)
    c5.append(len(xdata))
    c2.append(r)
    R0.append(np.exp(r * 4))
    # c3.append(popt1[0])
    c4.append(k)
    f0 = a
    ff = odeint(diff, f0, x, args=(r, k))
    inff1 = np.zeros(len(ff[:, 0]))
    # inff1[0]=ff[:,0][1]-ff[:,0][0]
    # inff1[0]=ydata[0]
    inff1[0] = ydata[0]
    for ii in range(1, len(ff[:, 0])):
        # inf.append(f[:,0][ii]-f[:,0][ii-1])
        inff1[ii] = ff[:, 0][ii] - ff[:, 0][ii - 1]
    yOut1 = inff1
    residuals1 = ydata - yOut1
    ss_res1 = np.sum(residuals1 ** 2)
    ss_tot1 = np.sum((ydata - np.mean(ydata)) ** 2)
    r_squared1 = 1 - (ss_res1 / ss_tot1)
    RR.append(r_squared1)
    # yOut1 =(ave*popt1[1])/(1+popt1[0]*np.exp(-(xdata*popt1[1])))
    fig, ax = plt.subplots()
    plt.scatter(xdata1[1:], ydata1[1:], c='k', s=18, label='Real data')
    plt.plot(x[1:], yOut1[1:], 'r', linewidth=2, label='Simulated')
    plt.legend(loc='best')
    ax.set_ylabel('Number of cases in ' + str(item), fontsize=18)
    ax.set_xlabel('Days since confirmed cases first reached 30 a day', fontsize=18)
    ax.tick_params(width=2, direction="out")
    fig.savefig(item, bbox_inches='tight')
    plt.close()
    df = dftemp1
with open('Data_july28Final.csv', 'w+') as output:
    writer2 = csv.writer(output)
    for key, value in D.items():
        writer2.writerow([key, value])
df3 = pd.DataFrame({'country': c1, 'Days': c5})
writer3 = pd.ExcelWriter('Days.xlsx')
df3.to_excel(writer3, sheet_name='data fitting', index=False)
writer3.save()
writer3.close()
df1 = pd.DataFrame(
    {'country': c1, 'Days_since': c5, 'growth_rate': c2, 'carry capacity': c4, 'R squared': RR, 'R0': R0})
writer1 = pd.ExcelWriter('data_fitting_results_August11final.xlsx')
df1.to_excel(writer1, sheet_name='data fitting', index=False)
writer1.save()
writer1.close()

df4 = pd.DataFrame({'country': c1, 'daysince': c5})
writer4 = pd.ExcelWriter('Days_since.xlsx')
df4.to_excel(writer4, sheet_name='days', index=False)
writer4.save()
writer4.close()
sys.exit()

print(df1)
CC2 = np.mean(c2)
CC21 = np.std(c2)
CC22 = min(c2)
CC23 = np.mean(RR)
CC24 = np.std(RR)
CC25 = min(RR)
print('The mean growth rate is :', CC2)
print('The standard deviation of the  growth rate is :', CC21)
print('The min  growth rate is :', CC22)
print('The mean R squared is :', CC23)
print('The standard deviation of the  R squared is :', CC24)
print('The min  R squared is :', CC25)

sys.exit()

scale0 = [2.98, 1.91]
mean = ['China', 'South Korea']


def logistic_model1(x, a, b, c):
    return c / (1 + a * np.exp(-b * x))


for i in mean:
    dftemp = df
    df = df.loc[(df['Entity'] == i)]
    xdata = df.Dayssince100.values
    # print(scale0[mean.index(i)])
    # ydata=df.cases.values*scale0[mean.index(i)]
    # ydata=df.cases.values*scale0[mean.index(i)]
    ydata = df.cases.values
    popt, pcov = curve_fit(logistic_model1, xdata, ydata, p0=[2, 1, 2000])
    c1.append(i)
    c2.append(popt[1])
    c3.append(popt[0])
    c4.append(popt[2])
    temp.append(popt[2])
    yOut = (popt[2]) / (1 + popt[0] * np.exp(-(xdata * popt[1])))
    residuals = ydata - yOut
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    RR.append(r_squared)
    fig, ax = plt.subplots()
    plt.scatter(xdata, ydata, c='k', s=18, label='Real data')
    plt.plot(xdata, yOut, 'r', linewidth=2, label='Simulated')
    plt.legend(loc='best')
    ax.set_ylabel('Total number of cases in ' + str(i), fontsize=18)
    ax.set_xlabel('Days', fontsize=18)
    ax.tick_params(width=2, direction="out")
    fig.savefig(i, bbox_inches='tight')
    plt.close()
    df = dftemp
ave = ((temp[0] / c2[0]) + (temp[1] / c2[1])) / 2


def logistic_model(x, a, b, c):
    # return (ave*b)/(1+a*np.exp(-b*x))
    return c / (1 + a * np.exp(-b * x))


countries = ['Australia', 'Austria', 'Bahrain', 'Belgium', 'Canada', 'Chile', 'Czech Republic', 'Denmark', 'Estonia',
             'Finland', 'France', 'Germany', 'Hungary', 'Iceland', 'Iran', 'Ireland', 'Israel', 'Italy', 'Japan',
             'Latvia', 'Lithuania', 'Luxembourg', 'Malaysia', 'Mexico', 'Netherlands', 'New Zealand', 'Norway',
             'Portugal', 'Qatar', 'Singapore', 'Slovakia', 'Spain', 'Sweden', 'Switzerland', 'Turkey', 'United Kingdom',
             'United States']
scale1 = [1.84, 2.24, 3.10, 2.50, 1.85, 2.25, 2.89, 2.01, 2.15, 2.53, 2.20, 2.03, 2.71, 3.58, 3.54, 2.15, 2.82, 2.14,
          2.31, 1.84, 2.08, 3.31, 2.24, 2.29, 2.01, 3.61, 2.61, 2.90, 3.91, 1.00, 3.09, 2.23, 2.06, 2.01, 2.60, 3.11,
          1.99, 1.83]
for item in countries:
    dftemp1 = df
    df = df.loc[(df['Entity'] == item)]
    xdata = df.Dayssince100.values
    # ydata=df.cases.values*scale1[countries.index(item)]
    ydata = df.cases.values
    popt1, pcov1 = curve_fit(logistic_model, xdata, ydata, p0=[2, 1, 2000])
    c1.append(item)
    c2.append(popt1[1])
    c3.append(popt1[0])
    c4.append(popt1[2])
    yOut1 = popt1[2] / (1 + popt1[0] * np.exp(-(xdata * popt1[1])))
    residuals1 = ydata - yOut1
    ss_res1 = np.sum(residuals1 ** 2)
    ss_tot1 = np.sum((ydata - np.mean(ydata)) ** 2)
    r_squared1 = 1 - (ss_res1 / ss_tot1)
    RR.append(r_squared1)
    # yOut1 =(ave*popt1[1])/(1+popt1[0]*np.exp(-(xdata*popt1[1])))
    fig, ax = plt.subplots()
    plt.scatter(xdata, ydata, c='k', s=18, label='Real data')
    plt.plot(xdata, yOut1, 'r', linewidth=2, label='Simulated')
    plt.legend(loc='best')
    ax.set_ylabel('Total number of cases in ' + str(item), fontsize=18)
    ax.set_xlabel('Days', fontsize=18)
    ax.tick_params(width=2, direction="out")
    fig.savefig(item, bbox_inches='tight')
    plt.close()
    df = dftemp

df1 = pd.DataFrame({'country': c1, 'growth_rate': c2, 'initial value': c3, 'carry capacity': c4, 'R squared': RR})
