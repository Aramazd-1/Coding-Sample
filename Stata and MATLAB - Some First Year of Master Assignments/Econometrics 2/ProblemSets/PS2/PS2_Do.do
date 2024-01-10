/*===============================================================================
 Date: 03/17/2023
 Econometrics 2, LMEC
 
 Dariga Alpyspayeva
 James Daniel Foltz  
 Dario Alfredo De Falco
================================================================================
								Problem Set 2
================================================================================*/

clear all
set more off
capture log close

cd "/Users/dannyfoltz/Desktop/Econometrics 2/Problem Set 2/Foltz_Alpyspayeva_DeFalco"

*log using "PS2_Log", text replace   

use "/Users/dannyfoltz/Desktop/Econometrics 2/Problem Set 2/Foltz_Alpyspayeva_DeFalco/PS2.dta", clear

*===============================================================================
*                    Part 1: Data Exploration      
*===============================================================================

* Q1:

*** GRAPH 1


gen dealer_string = "Non-Dealer" if dealer == 0 
replace dealer_string = "Dealer" if dealer == 1

graph hbox photos, over(dealer_string, label(labsize(*0.5))) ytitle("Number of Photos") title("Box Plot: Number of Photos by Dealer / Non-Dealer")

histogram photos, by(dealer_string, title("Photos for Listing: Dealer vs. Non-Dealer")) normal

graph export Graph_1.png, replace

*===============================================================================

* Q2:

bysort _sofsoftwar_1: egen mean_photos_sw1 = mean(photos) if _sofsoftwar_1 == 1 & dealer == 1
bysort _sofsoftwar_2: egen mean_photos_sw2 = mean(photos) if _sofsoftwar_2 == 1 & dealer == 1
bysort _sofsoftwar_3: egen mean_photos_sw3 = mean(photos) if _sofsoftwar_3 == 1 & dealer == 1


* Creating new dataframe to use
frame copy default collapsed_data

frame change collapsed_data

collapse (mean) mean_photos_sw1 = mean_photos_sw1 mean_photos_sw2 = mean_photos_sw2 mean_photos_sw3 = mean_photos_sw3

*** GRAPH 2

graph bar mean_photos_sw1 mean_photos_sw2 mean_photos_sw3, yvaroptions(relabel(1 "Average Photos - auction123" 2 "Average Photos - carad" 3 "Average Photos - eBizAutos"))

graph export Graph_2.png, replace

*===============================================================================

* Q3:

* Switching back to data before collapsing

frame change default

*** GRAPH 3

graph box logbid1, over(photos, label(labsize(*0.5))) ytitle("Log Bid") title("Box Plot: Log Bid vs. Number of Photos")

graph export Graph_3.png, replace

*===============================================================================
*                    Part 2: Analysis      
*===============================================================================


* Q1:

* What is photos supposed to be correlated with in the error term???

regress logbid1 photos logmiles options logfdback negpct, vce(cluster sellerid)
estimates store reg_q1




*===============================================================================

* Q2:

* Use code starting from line 263!!!!!

*===============================================================================

* Q3:

* Line 284!!!!!


regress photos _sofsoftwar_2 //first stage
estimate store reg_q3_first //save estimates
predict x_hat //predictions

regress logbid1 x_hat logmiles options logfdback negpct, vce(cluster sellerid) //second stage with predictors
estimate store reg_q3_second


*===============================================================================

* Q4:

ivreg2 logbid1 (photos = _sofsoftwar_2) logmiles options logfdback negpct, endog(photos) cluster(sellerid)
estimates store reg_q2

*===============================================================================

* Q5:

*===============================================================================

* Q6:

*===============================================================================

*log close
