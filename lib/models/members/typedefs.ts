export const memberTypeDefs = `#graphql
  type Member {
    id: ID!
    user: User!
    role: String!
    status: String!
    joinedAt: String!
  }

  extend type Space {
    members(limit: Int, offset: Int): [Member!]
    myMembership: Member
  }

  extend type Mutation {
    joinSpace(spaceId: ID!): Member!
    leaveSpace(spaceId: ID!): Boolean!
    
    # Admin actions
    updateMemberRole(spaceId: ID!, userId: ID!, role: String!): Member!
    removeMember(spaceId: ID!, userId: ID!): Boolean!
    approveMember(spaceId: ID!, userId: ID!): Member!
  }
`;
