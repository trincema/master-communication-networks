% Create Satellite Scenario
startTime = datetime(2021,12,10,18,27,57); % 10 December 2021, 6:27:57 PM UTC
stopTime = startTime + hours(3);           % 10 December 2021, 9:27:57 PM UTC
sampleTime = 60;                           % Seconds
sc = satelliteScenario(startTime,stopTime,sampleTime,"AutoSimulate",false);

% Add Large Constellation of Satellites
% openExample('aero_satcom/MultiHopPathSelectionThroughLargeSatelliteConstellationExample');
sat = satellite(sc,"largeConstellation.tle");
numSatellites = numel(sat)

% Add Ground Stations
gsSource = groundStation(sc,42.3001,-71.3504, ... % Latitude and longitude in degrees
    "Name","Source Ground Station");
gsTarget = groundStation(sc,45.7489,21.2087, ...  % Latitude and longitude in degrees
    "Name","Target Ground Station");

% Determine Elevation Angles of Satellites with Respect to Ground Stations
% Calculate the scenario state corresponding to StartTime.
advance(sc);

% Retrieve the elevation angle of each satellite with respect to the ground
% stations.
[~,elSourceToSat] = aer(gsSource,sat);
[~,elTargetToSat] = aer(gsTarget,sat);

% Determine the elevation angles that are greater than or equal to 30
% degrees.
elSourceToSatGreaterThanOrEqual30 = (elSourceToSat >= 30)';
elTargetToSatGreaterThanOrEqual30 = (elTargetToSat >= 30)';

% Determine Best Satellite for Initial Access to Constellation
% Find the indices of the elements of elSourceToSatGreaterThanOrEqual30
% whose value is true.
trueID = find(elSourceToSatGreaterThanOrEqual30 == true);

% These indices are essentially the indices of satellites in sat whose
% elevation angle with respect to "Source Ground Station" is at least 30
% degrees. Determine the range of these satellites to "Target Ground
% Station".
[~,~,r] = aer(sat(trueID), gsTarget);

% Determine the index of the element in r bearing the minimum value.
[~,minRangeID] = min(r);

% Determine the element in trueID at the index minRangeID.
id = trueID(minRangeID);

% This is the index of the best satellite for initial access to the
% constellation. This will be the first hop in the path. Initialize a
% variable 'node' that stores the first two nodes of the routing - namely,
% "Source Ground Station" and the best satellite for initial constellation
% access.
nodes = {gsSource sat(id)};

% Determine Remaining Nodes in Path to Target Ground Station
earthRadius = 6378137;                                                   % meters
altitude = 500000;                                                       % meters
horizonElevationAngle = asind(earthRadius/(earthRadius + altitude)) - 90 % degrees
% Minimum elevation angle of satellite nodes with respect to the prior
% node.
minSatElevation = -15; % degrees

% Flag to specify if the complete multi-hop path has been found.
pathFound = false;

% Determine nodes of the path in a loop. Exit the loop once the complete
% multi-hop path has been found.
while ~pathFound
    % Index of the satellite in sat corresponding to current node is
    % updated to the value calculated as index for the next node in the
    % prior loop iteration. Essentially, the satellite in the next node in
    % prior iteration becomes the satellite in the current node in this
    % iteration.
    idCurrent = id;

    % This is the index of the element in elTargetToSatGreaterThanOrEqual30
    % tells if the elevation angle of this satellite is at least 30 degrees
    % with respect to "Target Ground Station". If this element is true, the
    % routing is complete, and the next node is the target ground station.
    if elTargetToSatGreaterThanOrEqual30(idCurrent)
        nodes = {nodes{:} gsTarget}; %#ok<CCAT> 
        pathFound = true;
        continue
    end

    % If the element is false, the path is not complete yet. The next node
    % in the path must be determined from the constellation. Determine
    % which satellites have elevation angle that is greater than or equal
    % to -15 degrees with respect to the current node. To do this, first
    % determine the elevation angle of each satellite with respect to the
    % current node.
    [~,els] = aer(sat(idCurrent),sat); 

    % Overwrite the elevation angle of the satellite with respect to itself
    % to be -90 degrees to ensure it does not get re-selected as the next
    % node.
    els(idCurrent) = -90; 

    % Determine the elevation angles that are greater than or equal to -15
    % degrees.
    s = els >= minSatElevation;

    % Find the indices of the elements in s whose value is true.
    trueID = find(s == true);

    % These indices are essentially the indices of satellites in sat whose
    % elevation angle with respect to the current node is greater than or
    % equal to -15 degrees. Determine the range of these satellites to
    % "Target Ground Station".
    [~,~,r] = aer(sat(trueID), gsTarget);

    % Determine the index of the element in r bearing the minimum value.
    [~,minRangeID] = min(r);

    % Determine the element in trueID at the index minRangeID.
    id = trueID(minRangeID);

    % This is the index of the best satellite among those in sat to be used
    % for the next node in the path. Append this satellite to the 'nodes'
    % variable.
    nodes = {nodes{:} sat(id)}; %#ok<CCAT>
end

% Determine Intervals When Calculated Path Can Be Used
sc.AutoSimulate = true;
ac = access(nodes{:});
ac.LineColor = "red";
intvls = accessIntervals(ac)

% Visualize Path
v = satelliteScenarioViewer(sc,"ShowDetails",false);
sat.MarkerSize = 6; % Pixels
campos(v,60,5);     % Latitude and longitude in degrees

% Play the scenario.
play(sc);