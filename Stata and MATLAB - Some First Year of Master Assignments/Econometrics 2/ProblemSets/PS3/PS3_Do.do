/*===============================================================================
 Date: 04/07/2023
 Econometrics 2, LMEC
 
 Dariga Alpyspayeva
 James Daniel Foltz  
 Dario Alfredo De Falco
================================================================================
								Problem Set 3
================================================================================*/

clear all
set more off
capture log close

cd "/Users/dannyfoltz/Desktop/Econometrics 2/Problem Set 3/Foltz_Alpyspayeva_DeFalco"
*log using "PS3_Log", text replace   

use "/Users/dannyfoltz/Desktop/Econometrics 2/Problem Set 3/Foltz_Alpyspayeva_DeFalco/PS3.dta", clear

*===============================================================================
*                    Part 1: LPM     
*===============================================================================

* Q1:

global Z "i.warlag lgdpenlag lpoplag mtnest i.ncontig i.oi i.nwstate i.instab polity2lag i.laamcarib i.ssafrica i.seasia"

regress war CF $Z
estimates store reg_CF

regress war ELF $Z
estimates store reg_ELF

regress war ChiSq $Z
estimates store reg_ChiSq

regress war CF ELF ChiSq $Z
estimates store reg_All

*** TABLE 1

esttab reg_CF reg_ELF reg_ChiSq reg_All using "esttab.tex", replace ///
star(* 0.10 ** 0.05 *** 0.01) b(a3) se(3) nonum nogaps ///
stats(N, fmt(0 3) labels("Observations")) /// 
mtitles("(1)" "(2)" "(3)" "(4)") ///
keep( CF ELF ChiSq)  ///
se /// // 
prehead("\begin{table}[htbp]" /// // 
		"\caption{Regression Table \label{bwght}}" ///
		"\centering \renewcommand*{\arraystretch}{1.2}" /// 
		"\begin{threeparttable}" /// 
		"\begin{tabular}{l*{4}{c}}" ///
		"\hline") ///
postfoot("\hline \hline" /// 
		"\end{tabular}" ///
		"\begin{tablenotes} \footnotesize" /// 
		"\item Note: * \(p<0.10\), ** \(p<0.05\), *** \(p<0.01\). Standard Errors are in parentheses" ///
		"\end{tablenotes}" ///
		"\end{threeparttable}" ///
		"\end{table}")

*===============================================================================

* Q2:

* See response in the answer pdf


*===============================================================================
*                    Part 2: Logit    
*===============================================================================

* Q1:
logit war CF $Z

logit war ELF $Z

logit war ChiSq $Z

logit war CF ELF ChiSq $Z
*===============================================================================

* Q2:

quietly:logit war CF $Z
eststo M_CF: margins, dydx(CF) atmeans post

quietly:logit war ELF $Z
eststo M_ELF: margins, dydx(ELF) atmeans post

quietly:logit war ChiSq $Z
eststo M_ChiSq: margins, dydx(ChiSq) atmeans post

quietly:logit war CF ELF ChiSq $Z
eststo M_ALL: margins, dydx(CF ELF ChiSq) atmeans post

*** TABLE 2

esttab M_CF M_ELF M_ChiSq M_ALL using "esttab2.tex", replace ///
star(* 0.10 ** 0.05 *** 0.01) b(a3) se(3) nonum nogaps ///
stats(N, fmt(0 3) labels("Observations")) /// 
mtitles("(1)" "(2)" "(3)" "(4)") ///
keep( CF ELF ChiSq)  ///
se /// //
prehead("\begin{table}[htbp]" /// // 
		"\caption{Margins \label{bwght}}" ///
		"\centering \renewcommand*{\arraystretch}{1.2}" /// 
		"\begin{threeparttable}" /// 
		"\begin{tabular}{l*{4}{c}}" ///
		"\hline") ///
postfoot("\hline \hline" /// 
		"\end{tabular}" ///
		"\begin{tablenotes} \footnotesize" /// 
		"\item Note: * \(p<0.10\), ** \(p<0.05\), *** \(p<0.01\). Standard Errors are in parentheses" ///
		"\end{tablenotes}" ///
		"\end{threeparttable}" ///
		"\end{table}")


*===============================================================================

* Q3:
* Wald test
quietly: logit war CF ELF ChiSq warlag lgdpenlag lpoplag mtnest ncontig oi nwstate instab polity2lag laamcarib ssafrica seasia
test laamcarib ssafrica seasia
* LR-test
quietly: logit war CF ELF ChiSq warlag lgdpenlag lpoplag mtnest ncontig oi nwstate instab polity2lag laamcarib ssafrica seasia
estimate store full

quietly: logit war CF ELF ChiSq warlag lgdpenlag lpoplag mtnest ncontig oi nwstate instab polity2lag 
lrtest full

*===============================================================================

* Q4:
logit war CF ELF ChiSq $Z
estimates store logit_All


drop if _est_logit_All == 0
estat summarize 

*save them in a matrix
matrix list r(stats)
matrix r = r(stats)

scalar parteff_nwstate1=logistic(_b[_cons]+ ///
 _b[CF]*r[2,1]+ ///
 _b[ELF]*r[3,1] + ///
 _b[ChiSq]*r[4,1] + ///
 _b[1.warlag]*r[5,1] + ///
 _b[lgdpenlag]*r[6,1] + ///
 _b[lpoplag]*r[7,1] + ///
 _b[mtnest]*r[8,1]+ ///
 _b[1.ncontig]*r[9,1] + ///
 _b[1.oi]*r[10,1]+ ///
 _b[1.nwstate]*1 + ///
 _b[1.instab]*r[12,1]+ ///
 _b[polity2lag]*r[13,1] + ///
 _b[1.laamcarib]*r[14,1] + ///
 _b[1.ssafrica]*r[15,1] + ///
 _b[1.seasia]*r[16,1])
display parteff_nwstate1
 scalar parteff_nwstate0=logistic(_b[_cons]+ ///
 _b[CF]*r[2,1]+ ///
 _b[ELF]*r[3,1] + ///
 _b[ChiSq]*r[4,1] + ///
 _b[1.warlag]*r[5,1] + ///
 _b[lgdpenlag]*r[6,1] + ///
 _b[lpoplag]*r[7,1] + ///
 _b[mtnest]*r[8,1]+ ///
 _b[1.ncontig]*r[9,1] + ///
 _b[1.oi]*r[10,1]+ ///
 _b[1.nwstate]*0 + ///
 _b[1.instab]*r[12,1]+ ///
 _b[polity2lag]*r[13,1] + ///
 _b[1.laamcarib]*r[14,1] + ///
 _b[1.ssafrica]*r[15,1] + ///
 _b[1.seasia]*r[16,1])
display parteff_nwstate0
display parteff_nwstate1-parteff_nwstate0

*===============================================================================

* Q5:

scalar parteff_ChiSq=logisticden(_b[_cons]+ ///
 _b[CF]*r[2,1]+ ///
 _b[ELF]*r[3,1] + ///
 _b[ChiSq]*r[4,1] + ///
 _b[1.warlag]*r[5,1] + ///
 _b[lgdpenlag]*r[6,1] + ///
 _b[lpoplag]*r[7,1] + ///
 _b[mtnest]*r[8,1]+ ///
 _b[1.ncontig]*r[9,1] + ///
 _b[1.oi]*r[10,1]+ ///
 _b[1.nwstate]*r[11,1] + ///
 _b[1.instab]*r[12,1]+ ///
 _b[polity2lag]*r[13,1] + ///
 _b[1.laamcarib]*r[14,1] + ///
 _b[1.ssafrica]*r[15,1] + ///
 _b[1.seasia]*r[16,1])*_b[ChiSq]
 display parteff_ChiSq







*log close
