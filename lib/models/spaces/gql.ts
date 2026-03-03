import { gql } from "graphql-tag";

// ============================================
// FRAGMENTS
// ============================================

export const SPACE_FRAGMENT = gql`
  fragment SpaceFields on Space {
    id
    name
    slug
    description
    image
    categories
    visibility
    joinPolicy
    createdAt
    isActive
    membersCount
    type
    stripeAccountEnabled
  }
`;

// ============================================
// QUERIES
// ============================================

export const GET_ALL_SPACES = gql`
  ${SPACE_FRAGMENT}
  query GetAllSpaces {
    spaces {
      ...SpaceFields
    }
  }
`;

export const GET_MY_SPACES = gql`
  ${SPACE_FRAGMENT}
  query GetMySpaces {
    mySpaces {
      ...SpaceFields
      myMembership {
        role
      }
    }
  }
`;

export const GET_SPACE = gql`
  ${SPACE_FRAGMENT}
  query GetSpace($id: ID, $slug: String, $membersLimit: Int) {
    space(id: $id, slug: $slug) {
      ...SpaceFields
      myMembership {
        id
        role
        status
        tier {
          id
          name
          price
          interval
        }
      }
      tiers {
        id
        name
        description
        price
        currency
        interval
        isActive
        spaceId
      }
      members(limit: $membersLimit) {
        id
        role
        status
        joinedAt
        tier {
          name
        }
        user {
          id
          name
          email
        }
      }
    }
  }
`;

export const GET_SPACE_RECOMMENDED_EVENTS = gql`
  query GetSpaceRecommendedEvents($spaceId: ID!, $limit: Int) {
    space(id: $spaceId) {
      id
      recommendedEvents(limit: $limit) {
        id
        title
        location
        startsAt
        endsAt
        attendeeCount
        categories
        price
        isPaid
        space {
          id
          name
          slug
        }
      }
    }
  }
`;

// ============================================
// MUTATIONS
// ============================================

export const CREATE_SPACE = gql`
  mutation CreateSpace($input: CreateSpaceInput!) {
    createSpace(input: $input) {
      id
      name
      slug
    }
  }
`;

export const UPDATE_SPACE = gql`
  ${SPACE_FRAGMENT}
  mutation UpdateSpace($id: ID!, $input: UpdateSpaceInput!) {
    updateSpace(id: $id, input: $input) {
      ...SpaceFields
    }
  }
`;

export const DELETE_SPACE = gql`
  mutation DeleteSpace($id: ID!) {
    deleteSpace(id: $id)
  }
`;

export const JOIN_SPACE = gql`
  mutation JoinSpace($spaceSlug: String!, $tierId: ID) {
    joinSpace(spaceSlug: $spaceSlug, tierId: $tierId) {
      id
      status
    }
  }
`;

export const LEAVE_SPACE = gql`
  mutation LeaveSpace($spaceId: ID!) {
    leaveSpace(spaceId: $spaceId)
  }
`;
