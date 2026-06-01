
gen ln_Revenue_real = ln(Revenue_real)

reg ln_Revenue_real ///
    miami2023 miami2024 ///                     // Messi effects
    i.team_id i.Year ///                        // team + year FE
    Points Playoff MarketSize PctCapacity, ///  // controls
    vce(cluster team_id)                        // clustered SEs

reg Revenue_real ///
    miami2023 miami2024 ///
    i.team_id i.Year ///
    Points Playoff MarketSize PctCapacity ///
    if Year != 2020, ///
    vce(cluster team_id)

reg ln_Revenue_real ///
    miami2023 miami2024 ///
    i.team_id i.Year ///
    Points Playoff MarketSize PctCapacity ///
    if Year != 2020, ///
    vce(cluster team_id)
