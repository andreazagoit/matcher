import { gql } from "graphql-tag";

// ============================================
// FRAGMENTS
// ============================================

export const POST_FRAGMENT = gql`
  fragment PostFields on Post {
    id
    content
    mediaUrls
    likesCount
    commentsCount
    createdAt
    author {
      id
      firstName
      lastName
    }
  }
`;

// ============================================
// QUERIES
// ============================================

export const GET_GLOBAL_FEED = gql`
  ${POST_FRAGMENT}
  query GetGlobalFeed($limit: Int, $offset: Int) {
    globalFeed(limit: $limit, offset: $offset) {
      ...PostFields
    }
  }
`;

export const GET_SPACE_FEED = gql`
  ${POST_FRAGMENT}
  query GetSpaceFeed($spaceId: ID!, $limit: Int, $offset: Int) {
    space(id: $spaceId) {
      feed(limit: $limit, offset: $offset) {
        ...PostFields
      }
    }
  }
`;

// ============================================
// MUTATIONS
// ============================================

export const CREATE_POST = gql`
  mutation CreatePost($spaceId: ID!, $content: String!, $mediaUrls: [String!]) {
    createPost(spaceId: $spaceId, content: $content, mediaUrls: $mediaUrls) {
      id
    }
  }
`;

export const DELETE_POST = gql`
  mutation DeletePost($postId: ID!) {
    deletePost(postId: $postId)
  }
`;
