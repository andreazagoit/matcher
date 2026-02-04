export const postTypeDefs = `#graphql
  type Post {
    id: ID!
    content: String!
    mediaUrls: [String!]
    likesCount: Int
    commentsCount: Int
    createdAt: String!
    author: User!
  }

  extend type Space {
    feed(limit: Int, offset: Int): [Post!]
  }

  extend type Mutation {
    createPost(spaceId: ID!, content: String!, mediaUrls: [String!]): Post!
    deletePost(postId: ID!): Boolean!
  }
`;
