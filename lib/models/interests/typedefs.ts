export const interestTypeDefs = `#graphql
  type UserInterest {
    tag: String!
    weight: Float!
  }

  extend type Query {
    """
    Get the authenticated user's interests with weights.
    """
    myInterests: [UserInterest!]!
  }

  extend type Mutation {
    """
    Set the user's declared interests (replaces previous declared tags).
    """
    updateMyInterests(tags: [String!]!): [UserInterest!]!
  }
`;
