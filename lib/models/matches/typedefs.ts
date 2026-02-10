export const matchTypeDefs = `#graphql
  extend type Query {
    """
    Get daily suggested matches for the current user
    """
    dailyMatches: [User!]!
  }
`;
