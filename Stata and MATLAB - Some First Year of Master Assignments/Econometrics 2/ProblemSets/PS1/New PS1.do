/*===============================================================================
 Date: 03/03/2023
 Econometrics 2, LMEC
 Dariga Alpyspayeva
 James Daniel Foltz  
 Dario Alfredo De Falco
================================================================================
								Problem Set 1
================================================================================*/

clear all
set more off
capture log close
log using "Alpyspayeva_Foltz_Falco", text replace   

cd "D:\Study\#Stata\Econometrics 2\PS1"
use PS1


*===============================================================================
*                    Part 1: Data Exploration      
*===============================================================================

*1:

* encode string variables

encode country, gen(country_code)
encode continent, gen(continent_code)

* define panel data set and describe data

xtset country_code year

xtdescribe

*===============================================================================

*2:

* Analyzing country

xttab country_code

* Dropping countries with less than 18 observations

bysort country_code: egen country_code_total = count(country_code)
drop if country_code_total < 18

* Checking results as expected

xttab country_code

*===============================================================================

*3:

* Collapsing the dataset to calculate the average and standard deviation

collapse (mean) avg_asylums = asylums avg_temp = temp (sd) sd_asylums = asylums sd_temp = temp, by(country_code)

* Sorting on average asylums in descending order

gsort -avg_asylums

* Listing the 10 countries with highest average asylum applications to the EU

list country_code avg_asylums in 1/10

*===============================================================================

*4:

* Sorting on average temperature

gsort + avg_temp

* Generating a twoway scatterplot of avg asylums and std dev temperature

twoway scatter avg_asylums sd_temp, xtitle("Std Dev Temp") ytitle("Avg Asylum Applications") title("Graph 1") xlabel(.2(.1).9, labsize(small)) ylabel(0(10000)55000, labsize(small))

* Generating a two scatterplot of countries with avg asylums > 20,000 and a std dev temp < 0.6

twoway scatter avg_asylums sd_temp if avg_asylums > 20000 & sd_temp > .6, xtitle("Std Dev Temp") ytitle("Avg Asylum Applications") title("Graph 1 - Avg. Asylums > 20,000 & Std. Dev. Temp > 0.6") xlabel(.2(.1).9, labsize(small)) ylabel(0(10000)55000, labsize(small)) mlabel(country_code)

*===============================================================================

*5:

* Please see the pdf answer sheet

*===============================================================================

*6:

* Obtaining summary statistics for the main variables of interest

xtsum

*===============================================================================
*                    Part 2: Analysis      
*===============================================================================

*1:
gen ln_asylums = ln(asylums)

regress ln_asylums temp, robust
estimates store POLS

margins, at(temp=(0(1)40))

marginsplot, x(temp) xtitle("Temperature") ytitle("Predicted Log Asylum Applications") ylabel(, angle(horizontal)) title("Graph 2: log Asylum Applications vs. Temperature")

*2 
regress ln_asylums c.temp##c.temp, robust
estimates store POLS2
margins, at(temp=(0(1)40))


marginsplot, x(temp) xtitle("Temperature") ytitle("Predicted Log Asylum Applications") ylabel(, angle(horizontal)) title("Graph 3: log Asylum Applications vs. Temperature")

*3
xtreg ln_asylums c.temp##c.temp i.year, fe robust
estimates store FE1
*4
xtreg ln_asylums c.temp##c.temp c.rain##c.rain minor_conflict major_conflict i.year, fe robust
estimates store FE2 
margins, at(temp=(0(1)40))
*5
marginsplot, x(temp) xtitle("Temperature") ytitle("Predicted Log Asylum Applications")  title("Graph 4: log Asylum Applications vs. Temperature SD")
*Table 1
esttab POLS POLS2 FE1 FE2, se title("Table 1") ///
drop(*.year) ///
nonumbers mtitles("POLS1" "POLS2" "FE1" "FE2" ) star(* 0.10 ** 0.05 *** 0.01)




