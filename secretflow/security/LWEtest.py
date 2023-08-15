import secretflow as sf
import numpy as np
sf.init(['U1', 'U2', 'U3', 'U4', 'U5', 'U6', 'U7', 'U8', 'U9', 'U10', 'U11', 'U12', 'U13', 'U14', 'U15'], address='local')
# 'U16', 'U17', 'U18', 'U19', 'U120'
U1 = sf.PYU('U1')
U2 = sf.PYU('U2')
U3 = sf.PYU('U3')
U4 = sf.PYU('U4')
U5 = sf.PYU('U5')
U6 = sf.PYU('U6')
U7 = sf.PYU('U7')
U8 = sf.PYU('U8')
U9 = sf.PYU('U9')
U10 = sf.PYU('U10')
U11 = sf.PYU('U11')
U12 = sf.PYU('U12')
U13 = sf.PYU('U13')
U14 = sf.PYU('U14')
U15 = sf.PYU('U15')
# U16 = sf.PYU('U16')
# U17 = sf.PYU('U17')
# U18 = sf.PYU('U18')
# U19 = sf.PYU('U19')
# U20 = sf.PYU('U20')
#, U16,  U17, U18, U19, U20
aggregator=sf.security.LWESecureAggregator(U1,[U1, U2, U3, U4, U5, U6, U7, U8,
                                               U9, U10, U11, U12, U13, U14, U15],
                                           dimension_red_rate=0.2,
                                           noise_scale=3.0,
                                           clip_bound=30.0
                                          )
a = U1(lambda : np.array([3.2,6.2,7.4,4.5,6.2,7.8,2.2,5.1,1.4,5.8]))()
b = U2(lambda :np.array([-3.3,6.5,7.7,5.8,6.9,6.7,5.7,8.9,8.8,2.3]))()
c = U3(lambda : np.array([4.5,5.6,6.6,7.6,5.7,7.7,4.2,5.8,6.8,8.2]))()
d = U4(lambda : np.array([1.1,3.2,2.5,5.8,5.9,6.8,1.7,2.0,5.2,4.1]))()
e = U5(lambda : np.array([1.1,3.1,2.2,5.3,5.4,6.4,1.3,2.5,5.2,4.1]))()
f = U6(lambda : np.array([1.1,3.1,2.2,5.3,5.4,6.4,1.3,2.5,5.2,4.1]))()
g = U7(lambda : np.array([1.1,3.1,2.2,5.3,5.4,6.4,1.3,2.5,5.2,4.1]))()
h = U8(lambda : np.array([1.1,3.1,2.2,5.3,5.4,6.4,1.3,2.5,5.2,4.1]))()
i = U9(lambda : np.array([1.1,3.1,2.2,5.3,5.4,6.4,1.3,2.5,5.2,4.1]))()
j = U10(lambda : np.array([1.1,3.1,2.2,5.3,5.4,6.4,1.3,2.5,5.2,4.1]))()
k = U11(lambda : np.array([1.1,3.1,2.2,5.3,5.4,6.4,1.3,2.5,5.2,4.1]))()
l = U12(lambda : np.array([1.1,3.1,2.2,5.3,5.4,6.4,1.3,2.5,5.2,4.1]))()
m = U13(lambda : np.array([1.1,3.1,2.2,5.3,5.4,6.4,1.3,2.5,5.2,4.1]))()
n = U14(lambda : np.array([1.1,3.1,2.2,5.3,5.4,6.4,1.3,2.5,5.2,4.1]))()
o = U15(lambda : np.array([1.1,3.1,2.2,5.3,5.4,6.4,1.3,2.5,5.2,4.1]))()
# p = U16(lambda : np.array([1.1,3.1,2.2,5.3,5.4,6.4,1.3,2.5,5.2,4.1]))()
# q = U17(lambda : np.array([1.1,3.1,2.2,5.3,5.4,6.4,1.3,2.5,5.2,4.1]))()
# r = U18(lambda : np.array([1.1,3.1,2.2,5.3,5.4,6.4,1.3,2.5,5.2,4.1]))()
# s = U19(lambda : np.array([1.1,3.1,2.2,5.3,5.4,6.4,1.3,2.5,5.2,4.1]))()
# t = U20(lambda : np.array([1.1,3.1,2.2,5.3,5.4,6.4,1.3,2.5,5.2,4.1]))()



sum = aggregator.sum([a, b,c,d,e,f,g,h,i,j,k,l,m,n,o])
print(sf.reveal(sum))






# a = U1(lambda : np.array([3,6,7,4,6,7,2,5,1,5]))()
# b = U2(lambda :np.array([-3,6,7,5,6,6,5,8,8,2]))()
# c = U3(lambda : np.array([4,5,6,7,5,7,4,5,6,8]))()
# d = U4(lambda : np.array([1,3,2,5,5,6,1,2,5,4]))()
# e = U5(lambda : np.array([1,3,2,5,5,6,1,2,5,4]))()
# f = U6(lambda : np.array([1,3,2,5,5,6,1,2,5,4]))()
# g = U7(lambda : np.array([1,3,2,5,5,6,1,2,5,4]))()
# h = U8(lambda : np.array([1,3,2,5,5,6,1,2,5,4]))()
# import secretflow as sf
# import numpy as np
# sf.init(['alice','bob','petter','william'],address='local')
# alice = sf.PYU('alice')  #
# bob = sf.PYU('bob')
# petter = sf.PYU('petter')
# william = sf.PYU('william')
# aggregator=sf.security.SecureAggregator(alice,[alice,bob,petter,william],
#                                         dimension_red_rate = 0.3,
#                                         noise_scale=5.0,
#                                         clip_bound=30.0,
#                                         threshold_t=3,
# )
# a = alice(lambda : np.array([-3,6,7,4,6,7,2,5,1,5]))()
# b = bob(lambda : np.array([3,6,-7,5,6,6,5,8,-8,2]))()
# c = petter(lambda : np.array([4,5,6,-7,5,7,4,5,6,8]))()
# d = william(lambda : np.array([1,3,2,5,5,-6,1,2,5,4]))()
# sum_a_b = aggregator.sum([a, b,c,d], axis=0)
# print(sf.reveal(sum_a_b))
# a = alice(lambda : np.array([-3.0,6.2,7.5,4.5,6.5,7.8,2.8,5.8,1.8,5.8]))()
# b = bob(lambda : np.array([3.8,6.4,7.6,5.3,-6.2,6.1,5.6,8.4,-8.5,2.5]))()
# c = petter(lambda : np.array([4.5,5.8,6.3,7.2,5.7,7.8,4.6,5.8,6.1,8.2]))()
# d = william(lambda : np.array([1.5,3.4,-2.8,5.8,5.3,6.5,1.7,2.2,5.8,4.7]))()