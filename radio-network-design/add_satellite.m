% Create a satellite scenario and add ground stations from latitudes and longitudes.
startTime = datetime(2020,5,1,11,36,0);
stopTime = startTime + days(1);
sampleTime = 60;
sc = satelliteScenario(startTime,stopTime,sampleTime);
lat = 45.7489;
lon = 21.2087;
gs = groundStation(sc,lat,lon, "Name", "Ground Station Timisoara");

% Add satellites using Keplerian elements.
semiMajorAxis = 10000000;
eccentricity = 0;
inclination = 10; 
rightAscensionOfAscendingNode = 0; 
argumentOfPeriapsis = 0; 
trueAnomaly = 0; 
sat = satellite(sc,semiMajorAxis,eccentricity,inclination, ...
        rightAscensionOfAscendingNode,argumentOfPeriapsis,trueAnomaly);

% Add access analysis to the scenario and obtain the table of intervals of
% access between the satellite and the ground station.
ac = access(sat,gs);
intvls = accessIntervals(ac);
play(sc);