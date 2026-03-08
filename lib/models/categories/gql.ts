import { gql } from "graphql-tag";

export const GET_CATEGORIES = gql`
  query GetCategories {
    categories {
      id
    }
  }
`;

export const GET_CATEGORY = gql`
  query GetCategory($id: String!, $eventsLimit: Int, $spacesLimit: Int) {
    category(id: $id) {
      id
      recommendedEvents(limit: $eventsLimit) {
        id
        title
        description
        location
        startsAt
        endsAt
        spaceId
        price
        currency
        isPaid
        attendeeCount
        maxAttendees
        categories
      }
      recommendedSpaces(limit: $spacesLimit) {
        id
        name
        slug
        description
        cover
        categories
        visibility
        joinPolicy
        membersCount
        stripeAccountEnabled
        createdAt
      }
      recommendedCategories(limit: 6)
    }
  }
`;

export const GET_RECOMMENDED_CATEGORIES = gql`
  query GetRecommendedCategories($limit: Int, $offset: Int) {
    me {
      id
      recommendedCategories(limit: $limit, offset: $offset) {
        id
      }
    }
  }
`;
