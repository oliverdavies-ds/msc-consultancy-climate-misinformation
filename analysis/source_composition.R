# Super claim composition by source group (Figure 7), with a chi squared test
# of independence and Cramer's V as the effect size.
#
# Run from the repository root.

library(tidyverse)

df <- read_csv("data/public/corpus_results.csv")

CODE <- "\\d+_\\d+_\\d+(?:_\\d+)?"

# One row per statement and claim pair, with the five think tanks pooled.
long <- df |>
  mutate(group = recode(source,
                        "IEA" = "Think Tank", "Net Zero Watch" = "Think Tank",
                        "GWPF" = "Think Tank", "Adam Smith Institute" = "Think Tank",
                        "Centre for Policy Studies" = "Think Tank"),
         code = str_extract_all(predicted_claims, CODE)) |>
  unnest(code) |>
  filter(code != "0_0_0") |>
  mutate(super = str_sub(code, 1, 1))

# Contingency table of source group by super claim.
tab <- table(long$group, long$super)
print(tab)

# Share of each group's claims falling under each super claim.
plotdat <- as.data.frame(tab) |>
  rename(group = Var1, super = Var2, n = Freq) |>
  group_by(group) |>
  mutate(pct = 100 * n / sum(n)) |>
  ungroup()

ggplot(plotdat, aes(x = super, y = pct, fill = group)) +
  geom_col(position = "dodge", colour = "black") +
  labs(title = "Figure 7: Claim Distribution by Source", x = "CARDS Super Claim",
       y = "Percentage of claims", fill = NULL) +
  theme(plot.title = element_text(hjust = 0.5))

ggsave("figures/claim_distribution.png", width = 8, height = 5)

# Combined share of the science scepticism super claims (1, 2, 3 and 5) in each group.
plotdat |>
  filter(super %in% c("1", "2", "3", "5")) |>
  group_by(group) |>
  summarise(sci_pct = sum(pct))

# Chi squared test: is the super claim mix independent of source group?
chi <- chisq.test(tab)
print(chi)

# Cramer's V
sqrt(chi$statistic / (sum(tab) * (min(dim(tab)) - 1)))

# Pearson residuals show which cells drive the association.
round(chi$residuals, 2)
