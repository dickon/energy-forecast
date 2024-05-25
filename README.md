# Energy cost forecasting and solar modelling

When you have a choice of energy tariffs for a building, and are considering fabric and equipment changes you can make,
it is useful to be able to simulate what might happen in the future under different scenarios.

Currently this repo is exclusively targetting my home's solar panels, batteries, domestic electricity consumption,
and gas and electric heating but I have made a start on generalising out site specific parameters. That's partly
to avoid putting sensitive material into this public git repository.

Code is in [forceast.py](forecast.py), which requires a site specific configuration file which is not included.

## Step 1 - modelling solar production

Electricity import and export prices vary for me which change every 30 minutes. My electricity and gas billing metering also
captures incoming and outgoing at 30 minute resolution. So to simulate this we need to know when during the day we'll get
energy, down to at least 30 minute resolution.

There are a number of data sources we can use:

1. The electricity company meter (UK SMETS2 in my case cloud polling) which produces a 30 minute time series.
2. Solar inverter monitoring (SolarEdge monitoring cloud polling, 15 minute resolution)
3. Clamp transformer current monitoring (Vue Emporia cloud polling, 5 second resolution)
4. Battery system monitoring (Tesla Gateway local network access, 3 second resolution)
5. Weather data. (openweathermap.org cloud polling via Home Assistant. Alternatively I could have installed my own weather sensors but I don’t see the point when high quality reasonably local data is reliably available for free)

Counting up the output for each day using by Emporia clamp transformer monitoring and, later via Tesla system monitoring, we have:

![solar actual](solar_actual.png).

So we see the expected seasonal trend, with peaks over 60kWh per day in the summer. We also see a lot of days that fall short; it has been very cloudy and wet in the period, perhaps more than the expected level for eastern England.

If we count up in 30 minute buckets and plot with time of day on the Y axis, and use colour for the amount of electricity in the 30 minute bucket for that day we have:


![time series](solartimes.png)

In this form you can see that most of my peak generation is afternoon, which makes sense for an east-south-east facing array. We can also see that some days are better than others.

I was curious about the best power level I've seen, for each position of the sun. So I examine time in half hour buckets for various sun position (solar azimuth and elevation), take the maximum and use an algorithm to use the nearest value when no data is recorded.

![solar model](solarmodel.png)

My solar panels are on a roof facing east south east, and the treeline is fairly high in my area. That may well mean that in some sun positions the panels are in snadow, which would account for some of the readings. Or it may be that we've never had clear skies in the first 15 months of operations with the sun in a certain position. So, I expect when rerunning this code later to get a few new record power levels which will fill out the plot. Other system swill get very different fingerprints. 

So, now I have some idea of the maximum for differnet sun positions, and this allows me to work out the maximum if the skies were clear on a daily absis. 

Since I installed the system there's been a lot of cloud. We can compare what we would get by integrating the azium/elevation chart against the
sun positions for a day, which gives us a maximum. This can be integrated for a given day to give an approximate upper bound on what we'd get if we have clear skies all day. This maximum for each day according to our maximums for solar position can be plotted against time and overlaid the actual output. I also bucket to the week, and include the weekly mean cloud cover (taken as percentage of time with no cloud cover).

![dailysolar](dailysolar.png)

The line on the top graph is closer to sinusoid now, and I expect repeating this in a few years will show a smoother sinusoid. You can also see we've consistenly had a lot of cloud.
Over the life of the system, for me the actual output is 60% of what we get on a clear day.

Using the model to fill in some gaps in the monitoring, I have 9.38MWh in the 365 days up to 25 April 2024. In comparison, the SolarEdge design tool gave an output of 10.9MWh on an average year. That seemms within range given the famous unpredictability of British weather; it will be interesting once I have multiple years of data to see how this tracks.

# Step 2 - Energy demand modelling

Time of day matters of electricty demand matters wen considering solar power. The utility company meter gives me 30 minute resolution, when it works.
I have a [Vue Emporia energy montiros](https://www.emporiaenergy.com/energy-monitors) and I can get high resolution information from them.


# Step 3 - Battery modelling

# Step 4 - Octopus tariff integration

# Results

For my house:

![overall results](run.png)

To make sense of this, here's some summary and context:

 - I had 11.7kW of solar panels installed in January 2023, then 27kW of batteries installed September 2023. 
 - So far, as of March 2024, my energy bills since Jan 2023 have been £1322, compared with £3418 if I'd done nothing. 
 - In the next twelve months I expect to pay about £600 for energy, compared with £3000 if I'd done nothing.   
 - The house has gas central heating, now used exclusively for radiators. Gas consumption is included in the actual data and simulations. 
   We use approximately 20,000 kWh of gas per year, currently at 7.31p/kWh, 27.47p a day standing charge, so that's about £1600 a year. 
   So without gas my electricity bill is about -£1000/year for now. I'm considering moving to heat pumps, probably air to air. That might
   half our heating cost, though saving £800/year represents a long paybaack period on an expensive air to air system, with no government help 
   (and to be fair, no VAT either). So the motivation is mainly to reduce CO2 and to provide comfort in the rare summer heatwaves.
- Hot water is handled by the system gas boiler up to March 2023, then immersion at various times of day, from March 2023 through October 2023 using
   solar power, and from October 2023 using overnight off peak electricity. This is because the gas boiler typically use 10kWh/day for hot water, and the 
   immersion heater only uses 3kWh/day. (I'm interested in hot water tank heat pumps, but the noise is a concern).
 - Electricity and gas was supplied up to February 2023 by Ovo Energy on a cheap fixed tariff deal (2.66p/kWh 24.79p standing charge gas, 15.59p/kWh 23.33p standing charge electricity) which is no longer available.
 - 2 Tesla Powerwall 2 batteries installed September 2023. 13.5kWh nominal each.
 - We switched to Octopus Flux in October 2023
 - The big negative cost spikes in winter 2023/24 are Octopus Savings Sessions payments, where effectively
   the output rate shoots up for 30 to 90 minutes.
 - The scenarios in the future cover everything from what would have happened without solar or batteries and using gas for hot water, through
   to various Octous tariff combinations. It looks like if I stay with one tariff over the year for simplicity than Flux is a big win. And, surprisingly,
   Octopus Agile with dumping out the batteries when prices go high in the winter isn't worth it for us since that means forgoing the Savings Sessions payments.
   This is because Savings Sessions payments are only made relatively to what you normally do over the last 10 working days. 


- Dr. Dickon Reed, last updated 15 March 2024.
- dickon@cantab.net
