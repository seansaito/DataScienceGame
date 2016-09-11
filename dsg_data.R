library(corrplot)
library(Hmisc)

x_train <- read.csv("X_train.csv", header=TRUE, sep=",")
y_train <- read.csv("Y_train.csv", header=TRUE, sep=",")

variables <- colnames(x_train)

# Data Extraction #
Converted <- y_train[,2]

CustomerKey <- x_train[,"CustomerMD5Key"]
SCID <- lapply(x_train[,"SCID"], as.character)
SCID1 <- unlist(SCID)
MaritalStatus <- x_train[,"FirstDriverMaritalStatus"]
CarMileage <- x_train[,"CarAnnualMileage"]
CarFuelID <- x_train[,"CarFuelId"]
CarUsageID <- x_train[,"CarUsageId"]
DriverAge <- x_train[,"FirstDriverAge"]
CarInsuredValue <- x_train[,"CarInsuredValue"]
CarAge <- x_train[,"CarAge"]
LicenseNumber <- x_train[,"FirstDriverDrivingLicenseNumberY"]
VoluntaryExcess <- x_train[,"VoluntaryExcess"]
ParkingID <- x_train[,"CarParkingTypeId"]
NoClaimYears <- x_train[,"PolicyHolderNoClaimDiscountYears"]
LicenseType <- x_train[,"FirstDriverDrivingLicenceType"]
ClaimLastYear <- x_train[,"CoverIsNoClaimDiscountSelected"]
DrivingEntitlement <- x_train[,"CarDrivingEntitlement"]
DemographicID <- x_train[,"SocioDemographicId"]
ResidencyArea <- x_train[,"PolicyHolderResidencyArea"]
NumConvictions <- x_train[,"AllDriversNbConvictions"]
RatedDriverNum <- x_train[,"RatedDriverNumber"]
Homeowner <- x_train[,"IsPolicyholderAHomeowner"]
CarMakeID <- x_train[,"CarMakeId"]
DaysPurchase <- x_train[,"DaysSinceCarPurchase"]
PolicyName <- lapply(x_train[,"NameOfPolicyProduct"], as.character)
PolicyName1 <- unlist(PolicyName)
AffinityID <- x_train[,"AffinityCodeId"]

# Preliminary Data Processing # 

### DISCRETE VARIABLES ###
# AffinityID, CarFuelID, CarMakeID, CarUsageID, ClaimLastYear, DemographicID, DrivingEntitlement, 
# LicenseNumber, LicenseType, MaritalStatus, ParkingID, PolicyName1, RatedDriverNum, 
# ResidencyArea, SCID1

# Comparison with Converted #
tbl.AffinityID <- table(AffinityID,Converted)
tbl.MaritalStatus <- table(MaritalStatus,Converted)


### CONTINUOUS VARIABLES ###
# CarAge, CarInsuredValue, CarMileage, ClaimLastYear, DaysPurchase, DriverAge, NoClaimYears, 
# NumConvictions, VoluntaryExcess
cv <- data.frame(CarAge, CarInsuredValue, CarMileage, ClaimLastYear, DaysPurchase, DriverAge, NoClaimYears, NumConvictions, VoluntaryExcess)
corrplot(cor(cv),
         method = "circle",
         tl.col = "black")
