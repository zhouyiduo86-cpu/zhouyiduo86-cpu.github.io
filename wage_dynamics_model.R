library(dplyr)
library(ggplot2)
library(tidyr)
library(broom)

data <- read.csv("cps_00002_csv.gz")

df_clean <- data %>%
  filter(AGE >= 25, AGE <= 60) %>%
  filter(!is.na(INCWAGE), !is.na(UHRSWORKLY), !is.na(WKSWORK1),
         !is.na(AGE),     !is.na(EDUC),        !is.na(SEX)) %>%
  filter(INCWAGE >= 2200) %>%
  mutate(
    annual_hours = UHRSWORKLY * WKSWORK1,
    hourly_wage  = INCWAGE / annual_hours
  ) %>%
  filter(!(hourly_wage > 450 & annual_hours < 100))

cat("Cleaned N:", nrow(df_clean), "\n")

q_low  <- quantile(df_clean$INCWAGE, 0.01)
q_high <- quantile(df_clean$INCWAGE, 0.99)

df_trim <- df_clean %>%
  filter(INCWAGE >= q_low, INCWAGE <= q_high) %>%
  mutate(
    logwage  = log(INCWAGE),
    loghwage = log(hourly_wage),
    educ5 = case_when(
      EDUC == 73               ~ "HS",
      EDUC %in% c(81,91,92,111) ~ "SomeCollege",
      EDUC == 123              ~ "College",
      EDUC %in% c(124,125)    ~ "Grad",
      TRUE                     ~ "ltHS"
    ),
    educ5 = relevel(factor(educ5), ref = "ltHS"),
    sex_label  = ifelse(SEX == 1, "Men", "Women"),
    educ_label = recode(educ5,
                        ltHS        = "< High School",
                        HS          = "High School",
                        SomeCollege = "Some College",
                        College     = "College",
                        Grad        = "Graduate"
    )
  )

cat("Trimmed N:", nrow(df_trim), "\n")

skewness_fn <- function(x) { x <- x[is.finite(x)]; m <- mean(x); s <- sd(x); mean(((x-m)/s)^3) }
kurtosis_fn <- function(x) { x <- x[is.finite(x)]; m <- mean(x); s <- sd(x); mean(((x-m)/s)^4) }

make_row <- function(df, label) {
  vars <- c("AGE","EDUC","SEX","STATEFIP","INCWAGE","UHRSWORKLY","WKSWORK1")
  stats <- lapply(vars, function(v) {
    x <- df[[v]]
    c(mean=mean(x,na.rm=T), median=median(x,na.rm=T), sd=sd(x,na.rm=T),
      min=min(x,na.rm=T),   max=max(x,na.rm=T),
      skew=skewness_fn(x),  kurt=kurtosis_fn(x),
      IQR=IQR(x,na.rm=T))
  })
  data.frame(Sample=label, N=nrow(df), t(unlist(stats)), check.names=FALSE)
}

summary_table <- bind_rows(
  make_row(data,     "Raw"),
  make_row(df_clean, "Cleaned"),
  make_row(df_trim,  "Trimmed 1%")
)
print(summary_table)

df_reg <- df_trim %>%
  mutate(
    AGE_f   = relevel(factor(AGE),     ref = "25"),
    SEX_f   = factor(SEX),
    STATE_f = factor(STATEFIP)
  )

m1 <- lm(logwage ~ AGE_f + EDUC + SEX_f + STATE_f, data = df_reg)
m1 <- lm(logwage ~ factor(AGE) + factor(EDUC) + SEX_f + STATE_f, data = df_reg)
summary(m1)
cat("M1 R-squared:", summary(m1)$r.squared, "\n")

tab_m1 <- tidy(m1) %>%
  filter(grepl("factor\\(AGE\\)(30|35|40|45|50|55)$", term) |
           grepl("factor\\(EDUC\\)123", term) |         # college
           grepl("factor\\(EDUC\\)(124|125)", term) |   # grad
           grepl("SEX_f", term))
print(tab_m1)

gender_gap_annual <- coef(m1)["SEX_f2"]
cat("Annual gender gap (log):", round(gender_gap_annual, 4), "\n")
college_coeff <- coef(m1)["factor(EDUC)123"]
cat("College premium vs ltHS (log):", round(college_coeff, 4), "\n")
age59_coeff <- coef(m1)["factor(AGE)59"]
cat("Seniority gap 59 vs 25 (log):", round(age59_coeff, 4), "\n")
state_coeffs <- coef(m1)[grepl("STATE_f", names(coef(m1)))]
cat("Highest wage state:", names(which.max(state_coeffs)), round(max(state_coeffs), 4), "\n")
cat("Lowest wage state:",  names(which.min(state_coeffs)), round(min(state_coeffs), 4), "\n")

df_int <- df_trim %>%
  mutate(
    AGE_f   = relevel(factor(AGE), ref = "25"),
    SEX_f   = factor(SEX),
    STATE_f = factor(STATEFIP)
  )

m2 <- lm(
  logwage ~ AGE_f + educ5 + SEX_f +
    AGE_f:educ5 + SEX_f:educ5 + SEX_f:AGE_f +
    STATE_f,
  data = df_int
)
cat("M2 R-squared:", round(summary(m2)$r.squared, 4), "\n")

baseline_state <- as.character(names(sort(table(df_int$STATEFIP), decreasing=TRUE))[1])

pred_m2 <- function(age, sex, educ) {
  newdata <- data.frame(
    AGE_f   = factor(as.character(age), levels = levels(df_int$AGE_f)),
    SEX_f   = factor(sex,  levels = levels(df_int$SEX_f)),
    educ5   = factor(educ, levels = levels(df_int$educ5)),
    STATE_f = factor(baseline_state, levels = levels(df_int$STATE_f))
  )
  predict(m2, newdata = newdata)
}

educ_levels <- levels(df_int$educ5)
seniority_results <- expand.grid(educ = educ_levels, sex = c("1","2"),
                                 stringsAsFactors = FALSE) %>%
  mutate(gap_55_30 = mapply(function(e, s) pred_m2(55, s, e) - pred_m2(30, s, e),
                            educ, sex))
print(seniority_results)

gender_gap_college <- data.frame(age = c(28, 42, 55)) %>%
  mutate(gap_W_minus_M = sapply(age, function(a)
    pred_m2(a, "2", "College") - pred_m2(a, "1", "College")))
print(gender_gap_college)

college_premium <- expand.grid(age = c(26, 48), sex = c("1","2"),
                               stringsAsFactors = FALSE) %>%
  mutate(premium_College_vs_HS = mapply(function(a, s)
    pred_m2(a, s, "College") - pred_m2(a, s, "HS"), age, sex))
print(college_premium)

m3 <- lm(loghwage ~ factor(AGE) + educ5 + SEX_f + STATE_f, data = df_int)
cat("M3 R-squared:", round(summary(m3)$r.squared, 4), "\n")
gender_gap_hourly <- coef(m3)["SEX_f2"]
cat("Hourly gender gap (log):", round(gender_gap_hourly, 4), "\n")

educ_order  <- c("< High School","High School","Some College","College","Graduate")
educ_colors <- c("#d73027","#fc8d59","#fee090","#91bfdb","#4575b4")

fig1_data <- df_trim %>%
  group_by(AGE, sex_label, educ_label) %>%
  summarise(mean_logwage = mean(logwage), .groups = "drop") %>%
  mutate(educ_label = factor(educ_label, levels = educ_order))

ggplot(fig1_data, aes(x = AGE, y = mean_logwage, color = educ_label)) +
  geom_line(size = 0.9) + geom_point(size = 1.2) +
  facet_wrap(~ sex_label) +
  scale_color_manual(values = educ_colors, name = "Education") +
  labs(title = "Figure 1: Age-Wage Profiles by Education and Gender (CPS 2012)",
       x = "Age", y = "Mean Log Annual Wage") +
  theme_bw(base_size = 11) +
  theme(legend.position = "bottom", strip.text = element_text(face = "bold"))
ggsave("fig1_age_wage_profiles.png", width = 10, height = 4.2, dpi = 150)

fig2_data <- df_trim %>%
  group_by(AGE, sex_label, educ_label) %>%
  summarise(mean_logwage = mean(logwage), .groups = "drop") %>%
  pivot_wider(names_from = sex_label, values_from = mean_logwage) %>%
  mutate(gap = Women - Men,
         educ_label = factor(educ_label, levels = educ_order))

ggplot(fig2_data, aes(x = AGE, y = gap, color = educ_label)) +
  geom_line(size = 1) + geom_point(size = 1.5) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black", size = 0.5) +
  scale_color_manual(values = educ_colors, name = "Education") +
  labs(title = "Figure 2: Gender Log Wage Gap by Age and Education (CPS 2012)",
       x = "Age", y = "Log Wage Gap (Women minus Men)") +
  theme_bw(base_size = 11) +
  theme(legend.position = "bottom")
ggsave("fig2_gender_gap_age.png", width = 8, height = 4.5, dpi = 150)

fig3_data <- df_trim %>%
  filter(educ5 %in% c("HS","College")) %>%
  group_by(AGE, sex_label, educ5) %>%
  summarise(mean_logwage = mean(logwage), .groups = "drop") %>%
  pivot_wider(names_from = educ5, values_from = mean_logwage) %>%
  mutate(premium = College - HS)

ggplot(fig3_data, aes(x = AGE, y = premium, color = sex_label)) +
  geom_line(size = 1.2) + geom_point(size = 1.5) +
  scale_color_manual(values = c(Men = "#2166AC", Women = "#D6604D"), name = "Gender") +
  labs(title = "Figure 3: College Premium by Age and Gender (CPS 2012)",
       x = "Age", y = "College Premium (Log Wage, College minus HS)") +
  theme_bw(base_size = 11) +
  theme(legend.position = "bottom")
ggsave("fig3_college_premium_age.png", width = 8, height = 4.5, dpi = 150)

fig4_data <- df_trim %>%
  group_by(AGE, sex_label) %>%
  summarise(annual = mean(logwage), hourly = mean(loghwage), .groups = "drop") %>%
  pivot_longer(c(annual, hourly), names_to = "measure", values_to = "wage") %>%
  pivot_wider(names_from = sex_label, values_from = wage) %>%
  mutate(gap = Women - Men,
         measure = recode(measure, annual = "Annual Wage", hourly = "Hourly Wage"))

ggplot(fig4_data, aes(x = AGE, y = gap, color = measure)) +
  geom_line(size = 1.2) + geom_point(size = 1.5) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  scale_color_manual(values = c("Annual Wage" = "#762A83", "Hourly Wage" = "#1B7837"),
                     name = "Wage Measure") +
  labs(title = "Figure 4: Robustness — Annual vs. Hourly Gender Wage Gap by Age (CPS 2012)",
       x = "Age", y = "Log Wage Gap (Women minus Men)") +
  theme_bw(base_size = 11) +
  theme(legend.position = "bottom")
ggsave("fig4_robustness_gender_gap.png", width = 8, height = 4.2, dpi = 150)

cat("\nAll done. Figures saved to working directory.\n")