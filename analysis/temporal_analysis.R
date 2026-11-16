# Temporal analysis of contrarian claims (Figures 9 to 13, A1 and A2, Tables 6 to 10).
# All series are aggregated by month.
#
# Run from the repository root. The trend and break analyses need only the
# public label file. The bigram analysis at the end needs statement text, so it
# runs only when the full corpus results are available locally.

library(tidyverse)
library(lubridate)
library(broom)
library(huxtable)
library(tidylo)
library(car)
library(tidytext)

full_path   <- "data/corpus/corpus_results.csv"
public_path <- "data/public/corpus_results.csv"

df <- read_csv(if (file.exists(full_path)) full_path else public_path)

CODE <- "\\d+_\\d+_\\d+(?:_\\d+)?"


# ---- Monthly counts ---------------------------------------------------------

# Frequency of each individual claim by month, excluding the no claim code.
m_code <- tibble(
  month = floor_date(as.Date(df$date_published), "month"),
  code  = str_extract_all(df$predicted_claims, CODE)
) |>
  unnest(code) |>
  filter(code != "0_0_0") |>
  count(month, code)

# Figure 10: the eight most frequent claims over time.
top <- m_code |> count(code, wt = n, sort = TRUE) |> slice_head(n = 8) |> pull(code)

plot1 <- ggplot(filter(m_code, code %in% top), aes(month, n, colour = code)) +
  geom_line() +
  labs(title = "Figure 10: Claims over time by specific CARDS taxonomies", x = "Time",
       y = "Contrarian Statement Type", colour = "Code") +
  theme(plot.title = element_text(hjust = 0.5))

ggsave("figures/monthly_individualclaim_by_source.png", plot = plot1, width = 8, height = 5)

df <- df %>% mutate(date = ymd(date_published))

# A statement is claim bearing if it has any code other than 0_0_0.
df <- df %>% mutate(claimbearing = !str_detect(predicted_claims, "0_0_0"))

# Pool the five think tanks into one group.
df <- df %>% mutate(group = dplyr::recode(source,
                               "IEA" = "Think Tank",
                               "Net Zero Watch" = "Think Tank",
                               "GWPF" = "Think Tank",
                               "Adam Smith Institute" = "Think Tank",
                               "Centre for Policy Studies" = "Think Tank"))

# Monthly number of claim bearing statements by source group. Months with no
# claims are filled with zero so they are not dropped from the regressions.
monthly_source <- df |>
  filter(claimbearing) |>
  mutate(month = floor_date(date, "month")) |>
  count(month, group) |>
  complete(month, group, fill = list(n = 0))

# Figure 9
b <- ggplot(monthly_source, aes(month, n, colour = group)) +
  geom_line() +
  labs(title = "Figure 9: Frequency of claim-bearing statements overtime by source", x = "Year",
       y = "Claim-bearing statements", colour = "Source") +
  theme(plot.title = element_text(hjust = 0.5))

ggsave("figures/monthly_claims_by_source.png", plot = b, width = 8, height = 5)

# Monthly frequency of each super claim (the first level of the code).
super <- m_code |>
  mutate(super = str_sub(code, 1, 1)) |>
  count(month, super, wt = n)

# Figure 11
a <- ggplot(super, aes(month, n, colour = super)) +
  geom_line() +
  labs(title = "Figure 11: Frequency of super-claims overtime", x = "Year",
       y = "Claim-bearing statements", colour = "Super-claim") +
  theme(plot.title = element_text(hjust = 0.5))
ggsave("figures/monthly_superclaims_by_source.png", plot = a, width = 8, height = 5)

# Each super claim's share of all claims made that month.
super_den <- super |>
  group_by(month) |>
  mutate(share = n / sum(n)) |>
  ungroup()


# ---- Linear trends ----------------------------------------------------------

# Every model regresses a monthly series on time in years, so the coefficient
# is read as the change per year.

# Table 6: frequency of claim bearing statements, one model per source group.
freq_src <- monthly_source |>
  group_split(group) |>
  setNames(sort(unique(monthly_source$group))) |>
  lapply(\(d) lm(n ~ I(as.numeric(month)/365.25), data = d))

huxreg(freq_src,
       coefs = c("Annual trend" = "I(as.numeric(month)/365.25)"),
       statistics = c("Months" = "nobs", "R²" = "r.squared"),
       error_format = "({std.error})") |>
  set_caption("Table 6: Trend in climate misinformation statement frequency by source") |>
  quick_html(file = "tables/freq_src.html")

# Table 7: density, the percentage of each group's statements that carry a claim.
monthly_source_rate <- df |>
  mutate(month = floor_date(date, "month")) |>
  group_by(month, group) |>
  summarise(rate = 100 * mean(claimbearing), .groups = "drop")

dens_soc <- monthly_source_rate |>
  group_split(group) |>
  setNames(sort(unique(monthly_source_rate$group))) |>
  lapply(\(d) lm(rate ~ I(as.numeric(month)/365.25), data = d))

huxreg(dens_soc,
       coefs = c("Annual trend" = "I(as.numeric(month)/365.25)"),
       statistics = c("Months" = "nobs", "R²" = "r.squared"),
       error_format = "({std.error})") |>
  set_caption("Table 7: Trend in climate misinformation statement density by source") |>
  quick_html(file = "tables/dens_soc.html")

# Table 8: frequency of each super claim.
freq_sup <- super |>
  group_split(super) |>
  setNames(paste("Super", sort(unique(super$super)))) |>
  lapply(\(d) lm(n ~ I(as.numeric(month)/365.25), data = d))

huxreg(freq_sup,
       coefs = c("Annual trend" = "I(as.numeric(month)/365.25)"),
       statistics = c("Months" = "nobs", "R²" = "r.squared"),
       error_format = "({std.error})") |>
  set_caption("Table 8: Trend in climate misinformation statement frequency by super-claim") |>
  quick_html(file = "tables/freq_sup.html")

# Table 9: share of each super claim.
dens_sup <- super_den |>
  group_split(super) |>
  setNames(paste("Super", sort(unique(super_den$super)))) |>
  lapply(\(d) lm(share ~ I(as.numeric(month)/365.25), data = d))

huxreg(dens_sup,
       coefs = c("Annual trend" = "I(as.numeric(month)/365.25)"),
       statistics = c("Months" = "nobs", "R²" = "r.squared"),
       error_format = "({std.error})") |>
  set_caption("Table 9: Trend in climate misinformation statement share by super-claim") |>
  quick_html(file = "tables/dens_sup.html")


# ---- Volatility by source ---------------------------------------------------

# Mean, standard deviation and coefficient of variation of monthly counts.
monthly_source |>
  group_by(group) |>
  summarise(mean = mean(n), sd = sd(n), cv = sd / mean)

# Levene's test for equal variances across groups. Counts are transformed with
# log(1 + n) because some months have zero claims.
leveneTest(log1p(n) ~ factor(group), data = monthly_source)

# Overall trend across all sources.
total <- monthly_source |> count(month, wt = n, name = "n")
summary(lm(n ~ I(as.numeric(month)/365.25), data = total))

# Trend by source group in tidy form.
monthly_source |>
  group_by(group) |>
  group_modify(~ tidy(lm(n ~ I(as.numeric(month)/365.25), data = .x)))


# ---- Table 10: shift in super claim share from January 2026 -----------------

break_date <- as.Date("2026-01-01")

super_labels <- c(
  "1" = "1 Not happening", "2" = "2 Not us",
  "3" = "3 Not bad",
  "4" = "4 Solutions",
  "5" = "5 Science",
  "6" = "6 Actors",
  "7" = "7 Fossil fuels")

# One model per super claim: monthly share regressed on a before/after indicator.
# The intercept is the mean share before the break and the coefficient is the change after it.
break_models <- list()

for (s in names(super_labels)) {
  dat <- super_den |>
    filter(super == s) |>
    mutate(period = if_else(month >= break_date, "after", "before"),
           period = factor(period, levels = c("before", "after")))
  break_models[[super_labels[[s]]]] <- lm(share ~ period, data = dat)
}

break_table <- huxreg(break_models,
       coefs = c("Before January 2026" = "(Intercept)", "After January 2026" = "periodafter"),
       statistics = c("Number of months" = "nobs", "R²" = "r.squared",
                       "Adjusted R²" = "adj.r.squared", "Residual SD" = "sigma"),
       error_format = "({std.error})") |>
  set_caption("Table 10: Change in super-claim share after January 2026")

quick_html(break_table, file = "tables/Break.html")


# ---- Bigram analysis around the break (needs statement text) ----------------

if ("statement" %in% names(df)) {

  # Bigram counts before and after the break for one super claim, stop words removed.
  bigram_counts <- function(super_claim) {
    df |>
      filter(str_detect(predicted_claims, paste0("\\b", super_claim, "_"))) |>
      mutate(period = if_else(date >= break_date, "after", "before")) |>
      unnest_tokens(bigram, statement, token = "ngrams", n = 2) |>
      separate(bigram, c("w1", "w2"), sep = " ") |>
      filter(!w1 %in% stop_words$word, !w2 %in% stop_words$word) |>
      unite(bigram, w1, w2, sep = " ") |>
      count(period, bigram)
  }

  # Super claim 7 (fossil fuels are good)
  bg <- bigram_counts("7")

  bg |> filter(period == "before") |> slice_max(n, n = 10)
  bg |> filter(period == "after") |> slice_max(n, n = 10)

  # Weighted log odds identify the bigrams most distinctive of each period.
  # Bigrams seen fewer than 10 times are excluded as too noisy.
  log_odds <- bg |>
    filter(n >= 10) |>
    bind_log_odds(period, bigram, n)

  # Figure 12
  log_odds |>
    filter(period == "before") |>
    slice_max(log_odds_weighted, n = 10) |>
    mutate(bigram = reorder(bigram, log_odds_weighted)) |>
    ggplot(aes(log_odds_weighted, bigram)) +
    geom_col(fill = "steelblue") +
    xlim(0, 7) +
    labs(title = "Figure 12: Weighted log odds of super-claim 7 bigrams before January 2026",
         x = "Weighted log odds", y = "Bigrams") +
    theme(plot.title = element_text(hjust = 0.5))
  ggsave("figures/log_odds_before_7.png", width = 10, height = 5)

  # Figure 13
  log_odds |>
    filter(period == "after") |>
    slice_max(log_odds_weighted, n = 10) |>
    mutate(bigram = reorder(bigram, log_odds_weighted)) |>
    ggplot(aes(log_odds_weighted, bigram)) +
    geom_col(fill = "orange") +
    xlim(0, 7) +
    labs(title = "Figure 13: Weighted log odds of super-claim 7 bigrams after January 2026",
         x = "Weighted log odds", y = "Bigrams") +
    theme(plot.title = element_text(hjust = 0.5))
  ggsave("figures/log_odds_after_7.png", width = 10, height = 5)

  # Which source groups account for super claim 7 in each period.
  df |>
    filter(str_detect(predicted_claims, "\\b7_")) |>
    mutate(period = if_else(date >= break_date, "after", "before")) |>
    count(period, group) |>
    group_by(period) |>
    mutate(pct = round(100 * n / sum(n), 1)) |>
    print()

  # Super claim 6 (actors are biased), same method
  bg6 <- bigram_counts("6")

  bg6 |> filter(period == "before") |> slice_max(n, n = 10)
  bg6 |> filter(period == "after") |> slice_max(n, n = 10)

  log_odds_6 <- bg6 |>
    filter(n >= 10) |>
    bind_log_odds(period, bigram, n)

  # Figure A1
  p_before_6 <- log_odds_6 |>
    filter(period == "before") |>
    slice_max(log_odds_weighted, n = 10) |>
    mutate(bigram = reorder(bigram, log_odds_weighted)) |>
    ggplot(aes(log_odds_weighted, bigram)) +
    geom_col(fill = "steelblue") +
    labs(title = "Figure A1: Weighted log odds of super-claim 6 bigrams before January 2026",
         x = "Weighted log odds", y = "Bigrams") +
    theme(plot.title = element_text(hjust = 0.5))
  ggsave("figures/log_odds_before_6.png", plot = p_before_6, width = 10, height = 5)

  # Figure A2
  p_after_6 <- log_odds_6 |>
    filter(period == "after") |>
    slice_max(log_odds_weighted, n = 10) |>
    mutate(bigram = reorder(bigram, log_odds_weighted)) |>
    ggplot(aes(log_odds_weighted, bigram)) +
    geom_col(fill = "orange") +
    labs(title = "Figure A2: Weighted log odds of super-claim 6 bigrams after January 2026",
         x = "Weighted log odds", y = "Bigrams") +
    theme(plot.title = element_text(hjust = 0.5))
  ggsave("figures/log_odds_after_6.png", plot = p_after_6, width = 10, height = 5)
}
