export const postTypeDefs = `#graphql
  type Post {
    id: ID!
    content: String!
    mediaUrls: [String!]
    createdAt: DateTime!
    author: User!
    space: Space!
  }

  extend type User {
    """Posts from all spaces the user is an active member of."""
    feed(limit: Int, offset: Int): [Post!]!
  }

  extend type Space {
    feed(limit: Int, offset: Int): [Post!]!
  }

  extend type Mutation {
    createPost(spaceId: ID!, content: String!, mediaUrls: [String!]): Post!
    deletePost(postId: ID!): Boolean!
  }
`;
