# Energy cost forecasting and solar modelling

When you have a choice of energy tariffs for a building, and are considering fabric and equipment changes you can make,
it is useful to be able to simulate what might happen in the future under different scenarios.

Currently this repo is exclusively targetting my home's solar panels, batteries, domestic electricity consumption,
and gas and electric heating but I have made a start on generalising out site specific parameters. That's partly
to avoid putting sensitive material into this public git repository.

## Step 1 - modelling solar production

Electricity import and export prices vary for me which change every 30 minutes. My electricity and gas billing metering also
captures incoming and outgoing at 30 minute resolution. So to simulate this we need to know when during the day we'll get
energy, down to at least 30 minute resolution.

There are a number of data sources we can use:

1. The electricity company meter (UK SMETS2 in my case) which produces a 30 minute time series.
2. Solar inverter monitoring (e.g. SolarEdge monitoring, 15 minute resolution)
3. Clamp transformer current monitoring (Vue Emporia, 5 second resolution)
4. Battery system monitoring (e.g. Tesla Gateway for me, 3 second resolution)
5. Weather data.

Couting up the output for each day using by clamp transformer monitoring and, when I got it, the battery system monitoring, we have:

![solar actual](solar_actual.png).


Aggregating into 30 minute buckets and showing time of day on the Y axis, we have:


![time series](solartimes.png)

The structure becomes more self evident.

I then work out the peak solar output we've seen in a 30 minute period for solar azimuth and elevation, and fill in gaps using the nearest point we have:

![solar model](solarmodel.png)

My solar panels are on a roof facing east south east, and the treeline is fairly high in my area. Other system will get very different fingerprints.

So that's the maximum. Since I installed the system there's been a lot of cloud. We can compare what we would get by integrating the azium/elevation chart against the
sun positions for a day, which gives us a maximum. This can be integrated for a given day to give an approximate upper bound on what we'd get if we have clear skies all day.
This can then be plotted against the output, and bucketted to the week, to give:

![dialysolar](dailysolar.png)

# Step 2 - Energy  demand modelling

# Step 3 - Battery modelling

# Step 4 - Octopus tariff integration

# Results

For my house:

![overall results](run.png)

To make sense of this, here's some context:

 - The house has gas central heating, used for radiators. Hot water is handled by immersion at various times of day, from March through October using
   solar power, and from October 2023 using overnight off peak electricity. This is because the gas boiler typically use 10kWh/day for hot water, and the 
   immersion heater only uses 3kWh/day. 
 - 11.7kW of solar panels installed in January 2023
 - Electricity and gas supplied up to February 2024 
 - 2 Tesla Powerwall 2 batteries installed September 2023
 - Switched to Octopus Flux in October 2023
 - The big negative cost spikes in winter 2023/24 are Octopus Savings Sessions payments, where effectively
   the output rate shoots up for 30 to 90 minutes.
 - The scenarios in the future cover everything from what would have happened without solar or batteries and using gas for hot water, through
   to various Octous tariff combinations. It looks like if I stay with one tariff over the year for simplicity than Flux is a big win. And, surprisingly,
   Octopus Agile with dumping out the batteries when prices go high in the winter isn't worth it for us since that means forgoing the Savings Sessions payments.
   This is because Savings Sessions payments are only made relatively to what you normally do over the last 10 working days. 
