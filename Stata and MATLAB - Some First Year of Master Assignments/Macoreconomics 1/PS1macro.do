clear all
set more off
capture log close
cd "D:\Study\#Stata\Macoreconomics 1"
use data_ps1
drop country
encode countrycode, gen(country_code)

**dependent variable: 
gen gdp_pc= cgdpo/pop    //generate gdp per capita
gen ln_gdp_per_capita=  log(gdp_pc)  //log gdp per capita 
**independent variables:
bysort countrycode: egen avg_sk=mean(sk/100) //average 
gen ln_savings_capital=log(avg_sk)   //1st regressor

bysort countrycode: gen Diff_pop=pop-pop[_n-1] //difference by country between two consecutive years
bysort countrycode: gen n_pop=Diff_pop/pop[_n-1]
bysort countrycode: egen n=mean(n_pop)  //n as the average population growth rate for each country 

bysort countrycode: egen d=mean(delta) //average delta

gen ln_total_dep= log(n + 0.02 + d) //2nd regressor 
// Sort the data by country code and year
gen pop1519= 0.5*pop1519f+0.5*pop1519m  //population between 15 and 19 years; assuming 50% of males and 50% of females
gen humcap= ((sscenrol/100)*(pop1519/100))    //proxy with school enrollment
bysort countrycode: egen av_humcap=mean(humcap)

gen ln_savings_human=log(av_humcap) //new regressor
gen ln_human_capital=log(hc)
gen res_k=(ln_savings_capital - ln_total_dep)
gen res_h=(ln_human_capital - ln_total_dep)


// Keep only the observations for the year with the least missing values
keep if year == "2017"
// Actual regression
regress ln_gdp_per_capita ln_savings_capital ln_savings_human ln_total_dep, robust
estimates store Model1

regress ln_gdp_per_capita ln_savings_capital ln_human_capital ln_total_dep, robust
estimates store Model2
// Restricted regression
reg ln_gdp_per_capita res_k res_h

esttab Model1 Model2, se title("Table 1") ///
nonumbers mtitles("Augmented Solow" "Augmented Solow 2") star(* 0.10 ** 0.05 *** 0.01) ///

