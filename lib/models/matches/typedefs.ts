export const matchTypeDefs = `#graphql
  type DailyMatch {
    user: User!
    score: Float!
    distanceKm: Float
  }

  extend type User {
    """
    Today's 8 pre-computed matches based on bidirectional embedding similarity.
    Generates on-the-fly for the first request of the day, then cached in DB.
    Only visible to the authenticated user on their own profile.
    """
    dailyMatches: [DailyMatch!]!
  }
`;
