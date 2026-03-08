export const memberTypeDefs = `#graphql
  enum MemberRole {
    owner
    admin
    member
  }

  enum MemberStatus {
    pending
    waiting_payment
    active
    suspended
  }

  type Member {
    id: ID!
    user: User!
    role: MemberRole!
    status: MemberStatus!
    joinedAt: DateTime!
    subscriptionId: String
    currentPeriodEnd: DateTime
  }

  extend type Space {
    members(limit: Int, offset: Int): [Member!]!
    myMembership: Member
  }

  extend type Mutation {
    """Join a space by its slug. Optionally select a paid tier."""
    joinSpace(spaceSlug: String!, tierId: ID): Member!
    leaveSpace(spaceId: ID!): Boolean!

    # Admin actions
    updateMemberRole(spaceId: ID!, userId: ID!, role: MemberRole!): Member!
    removeMember(spaceId: ID!, userId: ID!): Boolean!
    approveMember(spaceId: ID!, userId: ID!): Member!
  }
`;
