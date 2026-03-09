
export const userItemTypeDefs = `#graphql
  """
  A single item on a user's profile — either a photo or a prompt answer.
  """
  type UserItem {
    id: ID!
    userId: ID!
    type: UserItemType!
    promptKey: String
    content: String!
    displayOrder: Int!
    createdAt: DateTime!
    updatedAt: DateTime!
  }

  enum UserItemType {
    photo
    prompt
  }

  extend type User {
    userItems: [UserItem!]!
  }
`;
